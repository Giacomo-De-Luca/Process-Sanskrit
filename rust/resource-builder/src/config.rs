use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

pub const CONFIG_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuilderConfig {
    pub schema_version: u32,
    #[serde(default)]
    pub generation: Option<GenerationConfig>,
    pub inputs: InputPaths,
    pub output: OutputConfig,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationConfig {
    pub legacy_data: PathBuf,
    pub neutral_output: PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputPaths {
    pub forms: PathBuf,
    pub sandhi_rules: PathBuf,
    pub scorer_metadata: PathBuf,
    pub syn0: PathBuf,
    pub syn1: PathBuf,
    pub code_offsets: PathBuf,
    pub code_data: PathBuf,
    pub point_offsets: PathBuf,
    pub point_data: PathBuf,
    pub log_table: PathBuf,
    pub sentencepiece_model: PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputConfig {
    pub directory: PathBuf,
}

impl BuilderConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path)
            .with_context(|| format!("failed to read resource config {}", path.display()))?;
        let mut config: Self = toml::from_str(&contents)
            .with_context(|| format!("invalid resource config {}", path.display()))?;
        if config.schema_version != CONFIG_SCHEMA_VERSION {
            bail!(
                "unsupported resource config schema {}; expected {}",
                config.schema_version,
                CONFIG_SCHEMA_VERSION
            );
        }

        let base = path.parent().unwrap_or_else(|| Path::new("."));
        config.resolve_paths(base);
        Ok(config)
    }

    fn resolve_paths(&mut self, base: &Path) {
        fn resolve(base: &Path, path: &mut PathBuf) {
            if path.is_relative() {
                *path = base.join(&*path);
            }
        }

        resolve(base, &mut self.inputs.forms);
        resolve(base, &mut self.inputs.sandhi_rules);
        resolve(base, &mut self.inputs.scorer_metadata);
        resolve(base, &mut self.inputs.syn0);
        resolve(base, &mut self.inputs.syn1);
        resolve(base, &mut self.inputs.code_offsets);
        resolve(base, &mut self.inputs.code_data);
        resolve(base, &mut self.inputs.point_offsets);
        resolve(base, &mut self.inputs.point_data);
        resolve(base, &mut self.inputs.log_table);
        resolve(base, &mut self.inputs.sentencepiece_model);
        resolve(base, &mut self.output.directory);
        if let Some(generation) = &mut self.generation {
            resolve(base, &mut generation.legacy_data);
            resolve(base, &mut generation.neutral_output);
        }
    }
}
