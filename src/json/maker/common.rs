use obfstr::obfstr;

use serde::Serialize;

use colored::Colorize;
use log::{info, debug, trace};
use rayon::prelude::*;

use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::error::Error;

use zip::write::{SimpleFileOptions, ZipWriter};

extern crate zip;
use crate::args::{Options, RUSTHOUND_VERSION};
use crate::objects::common::{Meta, LdapObject};

/// Current Bloodhound version 4.3+
pub const BLOODHOUND_VERSION_4: i8 = 6;

/// Number of objects serialized per parallel batch.
///
/// Objects in a batch are serialized to JSON concurrently (using all CPU
/// cores) then flushed sequentially to the output, so peak extra memory is
/// bounded by one batch instead of the whole dataset.
const SERIALIZE_BATCH: usize = 4096;

/// Where a JSON collection is written: either straight into an entry of the
/// shared zip archive, or into its own `.json` file on disk.
enum Sink<'a> {
    Zip { writer: &'a mut ZipWriter<BufWriter<File>>, datetime: &'a str, domain: &'a str },
    Dir { path: &'a str, datetime: &'a str, domain: &'a str },
}

/// Stream one object collection as a BloodHound JSON document
/// (`{"data":[...],"meta":{...}}`) into `writer`, without ever materialising
/// the whole document (or a `serde_json::Value` DOM) in memory.
///
/// Objects are serialized in parallel batches, which is both faster on
/// many-core machines and keeps memory flat regardless of object count.
fn write_json_document<T, W>(writer: &mut W, name: &str, vec_json: &[T]) -> std::io::Result<()>
where
    T: LdapObject + Serialize + Sync,
    W: Write,
{
    let count = vec_json.len();

    writer.write_all(b"{\"data\":[")?;

    let mut first = true;
    for batch in vec_json.chunks(SERIALIZE_BATCH) {
        // Serialize every object of this batch concurrently across all cores.
        let parts: Vec<String> = batch
            .par_iter()
            .map(|object| serde_json::to_string(object).expect("object serialization failed"))
            .collect();

        for part in parts {
            if !first {
                writer.write_all(b",")?;
            }
            first = false;
            writer.write_all(part.as_bytes())?;
        }
    }

    let meta = Meta::new(
        0_i32,
        name.to_owned(),
        count as i32,
        BLOODHOUND_VERSION_4,
        format!("RustHound-CE v{}", RUSTHOUND_VERSION.to_owned()),
    );
    writer.write_all(b"],\"meta\":")?;
    serde_json::to_writer(&mut *writer, &meta)?;
    writer.write_all(b"}")?;

    Ok(())
}

/// Emit one object collection to the selected [`Sink`]. Empty collections are
/// skipped, matching the previous behaviour (no file/entry is created).
fn emit<T>(sink: &mut Sink, name: &str, vec_json: &[T]) -> Result<(), Box<dyn Error>>
where
    T: LdapObject + Serialize + Sync,
{
    if vec_json.is_empty() {
        return Ok(());
    }

    let count = vec_json.len();
    debug!("Making {name}.json");

    match sink {
        Sink::Zip { writer, datetime, domain } => {
            // `large_file(true)` enables ZIP64, which is required for any entry
            // larger than 4 GiB. Without it the zip crate errors out on huge
            // domains, which is what caused the crash on large outputs.
            let options = SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated)
                .large_file(true);
            // Keep the original entry naming scheme: <datetime>_<domain>_<name>.json
            let filename = format!("{datetime}_{domain}_{name}.json");
            trace!("Adding file {}", filename.bold());
            writer.start_file(&filename, options)?;
            write_json_document(*writer, name, vec_json)?;
        }
        Sink::Dir { path, datetime, domain } => {
            let final_path = format!("{path}/{datetime}_{domain}_{name}.json");
            let file = File::create(&final_path)?;
            let mut buf = BufWriter::new(file);
            write_json_document(&mut buf, name, vec_json)?;
            buf.flush()?;
            info!("{} created!", final_path.bold());
        }
    }

    info!("{} {name} parsed!", count.to_string().bold());
    Ok(())
}

/// ACE right names RustHound-CE emits that are not part of BloodHound CE's
/// built-in AD schema. BloodHound drops these from the classic `Aces` arrays
/// ("unknown kind"), so they are also written as OpenGraph edges, which
/// BloodHound CE ingests with arbitrary edge kinds.
pub const OPENGRAPH_EDGE_KINDS: &[&str] = &["WriteRDN", "WriteManagedBy"];

/// Collect OpenGraph edges (as JSON values) for every ACE on `objects` whose
/// right name is in [`OPENGRAPH_EDGE_KINDS`]. Edges point from the trustee to
/// the object, both matched by BloodHound objectid.
fn collect_opengraph_edges<T: LdapObject>(objects: &[T], edges: &mut Vec<serde_json::Value>) {
    for object in objects {
        let target = object.get_object_identifier();
        for ace in object.get_aces() {
            let kind = ace.right_name().as_str();
            if !OPENGRAPH_EDGE_KINDS.contains(&kind) || ace.principal_sid() == target {
                continue;
            }
            edges.push(serde_json::json!({
                "kind": kind,
                "start": { "match_by": "id", "value": ace.principal_sid() },
                "end": { "match_by": "id", "value": target },
                "properties": { "isinherited": *ace.is_inherited() },
            }));
        }
    }
}

/// Build the OpenGraph document for the custom edge kinds, or `None` when
/// there are no such edges.
pub fn build_opengraph(ad: &crate::api::ADResults) -> Option<serde_json::Value> {
    let mut edges = Vec::new();
    collect_opengraph_edges(&ad.users, &mut edges);
    collect_opengraph_edges(&ad.groups, &mut edges);
    collect_opengraph_edges(&ad.computers, &mut edges);
    collect_opengraph_edges(&ad.ous, &mut edges);
    collect_opengraph_edges(&ad.gpos, &mut edges);
    if edges.is_empty() {
        return None;
    }
    Some(serde_json::json!({ "graph": { "nodes": [], "edges": edges } }))
}

/// Emit the OpenGraph document (see [`build_opengraph`]) to the given sink.
fn emit_opengraph(sink: &mut Sink, ad: &crate::api::ADResults) -> Result<(), Box<dyn Error>> {
    let Some(doc) = build_opengraph(ad) else {
        return Ok(());
    };
    let count = doc["graph"]["edges"].as_array().map_or(0, |e| e.len());
    let name = "opengraph";

    match sink {
        Sink::Zip { writer, datetime, domain } => {
            let options = SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated)
                .large_file(true);
            let filename = format!("{datetime}_{domain}_{name}.json");
            trace!("Adding file {}", filename.bold());
            writer.start_file(&filename, options)?;
            serde_json::to_writer(&mut **writer, &doc)?;
        }
        Sink::Dir { path, datetime, domain } => {
            let final_path = format!("{path}/{datetime}_{domain}_{name}.json");
            let mut buf = BufWriter::new(File::create(&final_path)?);
            serde_json::to_writer(&mut buf, &doc)?;
            buf.flush()?;
            info!("{} created!", final_path.bold());
        }
    }

    info!("{} OpenGraph edges ({}) parsed!", count.to_string().bold(), OPENGRAPH_EDGE_KINDS.join(", "));
    Ok(())
}

/// Emit every object collection contained in `ad_results` into the given sink.
macro_rules! emit_all {
    ($sink:expr, $ad:expr) => {{
        emit($sink, "users", &$ad.users)?;
        emit($sink, "groups", &$ad.groups)?;
        emit($sink, "computers", &$ad.computers)?;
        emit($sink, "ous", &$ad.ous)?;
        emit($sink, "domains", &$ad.domains)?;
        emit($sink, "gpos", &$ad.gpos)?;
        emit($sink, "containers", &$ad.containers)?;
        emit($sink, "ntauthstores", &$ad.ntauthstores)?;
        emit($sink, "aiacas", &$ad.aiacas)?;
        emit($sink, "rootcas", &$ad.rootcas)?;
        emit($sink, "enterprisecas", &$ad.enterprisecas)?;
        emit($sink, "certtemplates", &$ad.certtemplates)?;
        emit($sink, "issuancepolicies", &$ad.issuancepolicies)?;
        emit($sink, "sites", &$ad.sites)?;
        emit($sink, "siteservers", &$ad.siteservers)?;
        emit($sink, "sitesubnets", &$ad.sitesubnets)?;
        // Must come last: its edges reference nodes created by the files above.
        emit_opengraph($sink, $ad)?;
    }};
}

/// Write all collections as individual `.json` files on disk.
pub fn write_json_files(
    datetime: &str,
    domain_format: &str,
    common_args: &Options,
    ad_results: &crate::api::ADResults,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(&common_args.path)?;
    let mut sink = Sink::Dir {
        path: &common_args.path,
        datetime,
        domain: domain_format,
    };
    emit_all!(&mut sink, ad_results);
    Ok(())
}

/// Stream all collections into a single zip archive.
pub fn make_a_zip(
    datetime: &str,
    domain: &str,
    path: &str,
    ad_results: &crate::api::ADResults,
) -> Result<String, Box<dyn Error>> {
    fs::create_dir_all(path)?;
    let final_path = format!("{path}/{datetime}_{domain}_{}.zip", obfstr!("rusthound-ce"));

    let file = File::create(&final_path)?;
    // A large buffer keeps the single-threaded deflate stream fed and cuts
    // syscall overhead on multi-GB archives.
    let mut writer = ZipWriter::new(BufWriter::with_capacity(1 << 20, file));

    trace!("Making the ZIP file");
    {
        let mut sink = Sink::Zip { writer: &mut writer, datetime, domain };
        emit_all!(&mut sink, ad_results);
    }
    writer.finish()?.flush()?;

    info!("{} created!", &final_path.bold());
    Ok(final_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::common::AceTemplate;
    use crate::objects::group::Group;

    fn ace(sid: &str, right: &str, inherited: bool) -> AceTemplate {
        AceTemplate::new(sid.to_string(), "User".to_string(), right.to_string(), inherited, "".to_string())
    }

    #[test]
    fn opengraph_contains_only_custom_edge_kinds() {
        let mut group = Group::new();
        *group.object_identifier_mut() = "S-1-5-21-1-2-3-2000".to_string();
        group.set_aces(vec![
            ace("S-1-5-21-1-2-3-1105", "WriteManagedBy", false),
            ace("S-1-5-21-1-2-3-1106", "WriteRDN", true),
            ace("S-1-5-21-1-2-3-1107", "GenericAll", false),
            // Self-referencing ACEs are skipped, as BloodHound does.
            ace("S-1-5-21-1-2-3-2000", "WriteManagedBy", false),
        ]);
        let ad = crate::api::ADResults { groups: vec![group], ..Default::default() };

        let doc = build_opengraph(&ad).expect("edges expected");
        assert!(doc.get("meta").is_none(), "must not look like a classic collector file");
        let edges = doc["graph"]["edges"].as_array().unwrap();
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0]["kind"], "WriteManagedBy");
        assert_eq!(edges[0]["start"]["value"], "S-1-5-21-1-2-3-1105");
        assert_eq!(edges[0]["end"]["value"], "S-1-5-21-1-2-3-2000");
        assert_eq!(edges[0]["properties"]["isinherited"], false);
        assert_eq!(edges[1]["kind"], "WriteRDN");
        assert_eq!(edges[1]["properties"]["isinherited"], true);
    }

    #[test]
    fn opengraph_is_skipped_without_custom_edges() {
        let mut group = Group::new();
        group.set_aces(vec![ace("S-1-5-21-1-2-3-1107", "GenericAll", false)]);
        let ad = crate::api::ADResults { groups: vec![group], ..Default::default() };
        assert!(build_opengraph(&ad).is_none());
    }
}
