use std::path::PathBuf;

use anyhow::{bail, Result};
use process_sanskrit_resource_builder::{default_config_path, ResourceBuilder};

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let config = match args.next() {
        None => default_config_path(),
        Some(flag) if flag == "--config" => args
            .next()
            .map(PathBuf::from)
            .ok_or_else(|| anyhow::anyhow!("--config requires a path"))?,
        Some(other) => bail!("unknown argument {:?}; expected --config PATH", other),
    };
    if args.next().is_some() {
        bail!("unexpected extra arguments");
    }

    let manifest = ResourceBuilder::from_config_path(config)?.build()?;
    println!(
        "built {} forms, {} sandhi keys, and {} scorer vocabulary rows",
        manifest.counts.forms, manifest.counts.sandhi_keys, manifest.counts.word2vec_vocab
    );
    Ok(())
}
