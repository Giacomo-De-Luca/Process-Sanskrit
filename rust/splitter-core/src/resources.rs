use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use fst::Set;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{CoreError, DcsScorer, PieceEncoder, SandhiRules, ScorerModel, SequenceScorer};

const MANIFEST_NAME: &str = "native-assets.json";
const SCHEMA_VERSION: u32 = 1;

enum FormStore {
    Memory(BTreeSet<String>),
    Native(Set<Vec<u8>>),
}

struct VerifiedGraphAssets {
    forms: Vec<u8>,
    sandhi_after: Vec<u8>,
    sandhi_variants: Vec<u8>,
}

struct VerifiedScoredAssets {
    graph: VerifiedGraphAssets,
    scorer: Vec<u8>,
    scorer_vocab: Vec<u8>,
    sentencepiece_model: Vec<u8>,
}

/// Immutable resources shared by all request-local split graphs.
pub struct Resources {
    forms: FormStore,
    rules: SandhiRules,
    scorer: Option<Arc<dyn SequenceScorer>>,
}

impl std::fmt::Debug for Resources {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Resources")
            .field("rules", &self.rules)
            .field("has_scorer", &self.scorer.is_some())
            .finish_non_exhaustive()
    }
}

impl Resources {
    /// Construct small deterministic resources without package assets.
    pub fn from_memory(
        forms: BTreeSet<String>,
        rules: SandhiRules,
        scorer: Option<Arc<dyn SequenceScorer>>,
    ) -> Self {
        Self {
            forms: FormStore::Memory(forms),
            rules,
            scorer,
        }
    }

    /// Verify package assets before constructing a tokenizer from their bytes.
    pub fn load_with_encoder_factory<F>(
        data_dir: impl AsRef<Path>,
        create_encoder: F,
    ) -> Result<Self, CoreError>
    where
        F: FnOnce(&[u8]) -> Result<Arc<dyn PieceEncoder>, CoreError>,
    {
        let data_dir = data_dir.as_ref();
        let assets = Self::read_scored_assets(data_dir)?;
        let encoder = create_encoder(&assets.sentencepiece_model)?;
        Self::load_verified(assets, encoder)
    }

    fn read_scored_assets(data_dir: &Path) -> Result<VerifiedScoredAssets, CoreError> {
        let manifest = AssetManifest::load(data_dir)?;
        let graph = VerifiedGraphAssets::read(&manifest, data_dir)?;
        let scorer = manifest.read_verified(data_dir, "scorer.bin")?;
        let scorer_vocab = manifest.read_verified(data_dir, "scorer_vocab.fst")?;
        let sentencepiece_model = manifest.read_verified(data_dir, "sentencepiece.model")?;
        Ok(VerifiedScoredAssets {
            graph,
            scorer,
            scorer_vocab,
            sentencepiece_model,
        })
    }

    fn load_verified(
        assets: VerifiedScoredAssets,
        encoder: Arc<dyn PieceEncoder>,
    ) -> Result<Self, CoreError> {
        let forms = load_forms(assets.graph.forms)?;
        let rules =
            SandhiRules::from_owned_bytes(assets.graph.sandhi_after, assets.graph.sandhi_variants)?;
        let scorer_model = ScorerModel::from_owned_bytes(assets.scorer, assets.scorer_vocab)?;
        let scorer: Arc<dyn SequenceScorer> = Arc::new(DcsScorer::new(scorer_model, encoder));
        Ok(Self {
            forms: FormStore::Native(forms),
            rules,
            scorer: Some(scorer),
        })
    }

    /// Load graph resources without silently constructing a heuristic scorer.
    /// This is intended only for explicit `score=False` and diagnostic tests.
    pub fn load_unscored(data_dir: impl AsRef<Path>) -> Result<Self, CoreError> {
        let data_dir = data_dir.as_ref();
        let manifest = AssetManifest::load(data_dir)?;
        let assets = VerifiedGraphAssets::read(&manifest, data_dir)?;
        Ok(Self {
            forms: FormStore::Native(load_forms(assets.forms)?),
            rules: SandhiRules::from_owned_bytes(assets.sandhi_after, assets.sandhi_variants)?,
            scorer: None,
        })
    }

    pub fn valid(&self, word: &str) -> bool {
        match &self.forms {
            FormStore::Memory(forms) => forms.contains(word),
            FormStore::Native(forms) => forms.contains(word),
        }
    }

    pub fn rules(&self) -> &SandhiRules {
        &self.rules
    }

    pub fn scorer(&self) -> Result<&dyn SequenceScorer, CoreError> {
        self.scorer.as_deref().ok_or_else(|| {
            CoreError::ScorerUnavailable(
                "native resources were initialized without scorer.bin and SentencePiece".to_owned(),
            )
        })
    }
}

impl VerifiedGraphAssets {
    fn read(manifest: &AssetManifest, data_dir: &Path) -> Result<Self, CoreError> {
        Ok(Self {
            forms: manifest.read_verified(data_dir, "forms.fst")?,
            sandhi_after: manifest.read_verified(data_dir, "sandhi_after.fst")?,
            sandhi_variants: manifest.read_verified(data_dir, "sandhi_variants.bin")?,
        })
    }
}

fn load_forms(bytes: Vec<u8>) -> Result<Set<Vec<u8>>, CoreError> {
    Set::new(bytes).map_err(|error| CoreError::asset("forms.fst", error.to_string()))
}

#[derive(Debug, Deserialize)]
struct AssetManifest {
    schema_version: u32,
    assets: BTreeMap<String, AssetRecord>,
}

#[derive(Debug, Deserialize)]
struct AssetRecord {
    bytes: u64,
    sha256: String,
    format: String,
}

impl AssetManifest {
    fn load(data_dir: &Path) -> Result<Self, CoreError> {
        let path = data_dir.join(MANIFEST_NAME);
        let bytes = fs::read(&path).map_err(|error| CoreError::io(&path, error))?;
        let manifest: Self = serde_json::from_slice(&bytes)
            .map_err(|error| CoreError::asset(MANIFEST_NAME, error.to_string()))?;
        if manifest.schema_version != SCHEMA_VERSION {
            return Err(CoreError::asset(
                MANIFEST_NAME,
                format!(
                    "schema version {} is unsupported; expected {SCHEMA_VERSION}",
                    manifest.schema_version
                ),
            ));
        }
        Ok(manifest)
    }

    fn read_verified(&self, data_dir: &Path, name: &str) -> Result<Vec<u8>, CoreError> {
        let record = self.assets.get(name).ok_or_else(|| {
            CoreError::asset(MANIFEST_NAME, format!("asset entry {name:?} is missing"))
        })?;
        let expected_format = match name {
            "forms.fst" => "fst-set-v1",
            "sandhi_after.fst" => "fst-map-v1",
            "sandhi_variants.bin" => "process-sanskrit-sandhi-variants-v1",
            "scorer.bin" => "process-sanskrit-scorer-v1",
            "scorer_vocab.fst" => "fst-map-v1",
            "sentencepiece.model" => "sentencepiece-model-protobuf",
            _ => {
                return Err(CoreError::asset(
                    MANIFEST_NAME,
                    format!("loader has no format contract for {name:?}"),
                ));
            }
        };
        if record.format != expected_format {
            return Err(CoreError::asset(
                MANIFEST_NAME,
                format!(
                    "asset {name:?} uses format {:?}; expected {expected_format:?}",
                    record.format
                ),
            ));
        }
        let path = data_dir.join(name);
        let bytes = fs::read(&path).map_err(|error| CoreError::io(&path, error))?;
        if bytes.len() as u64 != record.bytes {
            return Err(CoreError::asset(
                name,
                format!(
                    "size is {} bytes but manifest requires {}",
                    bytes.len(),
                    record.bytes
                ),
            ));
        }
        let digest = sha256(&bytes);
        if !digest.eq_ignore_ascii_case(&record.sha256) {
            return Err(CoreError::ChecksumMismatch {
                asset: name.to_owned(),
            });
        }
        Ok(bytes)
    }
}

fn sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
