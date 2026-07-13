mod config;

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, ensure, Context, Result};
pub use config::{
    BuilderConfig, GenerationConfig, InputPaths, OutputConfig, CONFIG_SCHEMA_VERSION,
};
use fst::{MapBuilder, SetBuilder};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use prost::Message;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const ASSET_SCHEMA_VERSION: u32 = 1;
pub const FORMS_FILE: &str = "forms.fst";
pub const SANDHI_MAP_FILE: &str = "sandhi_after.fst";
pub const SANDHI_VARIANTS_FILE: &str = "sandhi_variants.bin";
pub const SCORER_FILE: &str = "scorer.bin";
pub const SCORER_VOCAB_FILE: &str = "scorer_vocab.fst";
pub const SENTENCEPIECE_FILE: &str = "sentencepiece.model";
pub const MANIFEST_FILE: &str = "native-assets.json";

const VARIANTS_MAGIC: &[u8; 8] = b"PSSV0001";
const SCORER_MAGIC: &[u8; 8] = b"PSSC0001";
const SCORER_HEADER_LEN: usize = 128;
const SECTION_ALIGNMENT: usize = 64;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuleGroup {
    after: String,
    variants: Vec<RuleVariant>,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd)]
#[serde(deny_unknown_fields)]
struct RuleVariant {
    left: String,
    right: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScorerMetadata {
    schema_version: u32,
    window: u32,
    cbow_mean: bool,
    vocab: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeAssetManifest {
    pub schema_version: u32,
    pub generator: GeneratorRecord,
    pub sources: BTreeMap<String, FileRecord>,
    pub assets: BTreeMap<String, AssetRecord>,
    pub counts: CountRecord,
    pub scorer: ScorerRecord,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeneratorRecord {
    pub name: &'static str,
    pub version: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct FileRecord {
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct AssetRecord {
    pub format: &'static str,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct CountRecord {
    pub forms: u64,
    pub sandhi_keys: u64,
    pub sandhi_variants: u64,
    pub word2vec_vocab: u64,
    pub sentencepiece_vocab: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ScorerRecord {
    pub vector_dimension: u32,
    pub window: u32,
    pub cbow_mean: bool,
    pub log_table_entries: u32,
}

pub struct ResourceBuilder {
    config: BuilderConfig,
}

impl ResourceBuilder {
    pub fn new(config: BuilderConfig) -> Self {
        Self { config }
    }

    pub fn from_config_path(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::new(BuilderConfig::from_path(path)?))
    }

    pub fn build(&self) -> Result<NativeAssetManifest> {
        let output = &self.config.output.directory;
        fs::create_dir_all(output)
            .with_context(|| format!("failed to create output directory {}", output.display()))?;
        let staging = output.join(".native-resource-build");
        if staging.exists() {
            fs::remove_dir_all(&staging).with_context(|| {
                format!("failed to clear staging directory {}", staging.display())
            })?;
        }
        fs::create_dir(&staging)
            .with_context(|| format!("failed to create staging directory {}", staging.display()))?;

        let result = self.build_in(&staging);
        if result.is_err() {
            let _ = fs::remove_dir_all(&staging);
            return result;
        }
        let manifest = result?;

        for name in [
            FORMS_FILE,
            SANDHI_MAP_FILE,
            SANDHI_VARIANTS_FILE,
            SCORER_FILE,
            SCORER_VOCAB_FILE,
            SENTENCEPIECE_FILE,
        ] {
            install_asset(&staging, output, name)?;
        }
        // The manifest is the commit marker: loaders reject partial or stale assets by hash.
        install_asset(&staging, output, MANIFEST_FILE)
            .context("failed to install native asset manifest")?;
        fs::remove_dir_all(&staging)
            .with_context(|| format!("failed to remove staging directory {}", staging.display()))?;
        Ok(manifest)
    }

    fn build_in(&self, staging: &Path) -> Result<NativeAssetManifest> {
        let forms = build_forms(&self.config.inputs.forms, &staging.join(FORMS_FILE))?;
        let (sandhi_keys, sandhi_variants) = build_sandhi(
            &self.config.inputs.sandhi_rules,
            &staging.join(SANDHI_MAP_FILE),
            &staging.join(SANDHI_VARIANTS_FILE),
        )?;
        let scorer = build_scorer(
            &self.config.inputs,
            &staging.join(SCORER_FILE),
            &staging.join(SCORER_VOCAB_FILE),
        )?;
        fs::copy(
            &self.config.inputs.sentencepiece_model,
            staging.join(SENTENCEPIECE_FILE),
        )
        .with_context(|| {
            format!(
                "failed to copy {}",
                self.config.inputs.sentencepiece_model.display()
            )
        })?;

        let mut sources = BTreeMap::new();
        for (name, path) in source_paths(&self.config.inputs) {
            sources.insert(name.to_owned(), file_record(path)?);
        }

        let mut assets = BTreeMap::new();
        for (name, format) in [
            (FORMS_FILE, "fst-set-v1"),
            (SANDHI_MAP_FILE, "fst-map-v1"),
            (SANDHI_VARIANTS_FILE, "process-sanskrit-sandhi-variants-v1"),
            (SCORER_FILE, "process-sanskrit-scorer-v1"),
            (SCORER_VOCAB_FILE, "fst-map-v1"),
            (SENTENCEPIECE_FILE, "sentencepiece-model-protobuf"),
        ] {
            let record = file_record(staging.join(name))?;
            assets.insert(
                name.to_owned(),
                AssetRecord {
                    format,
                    bytes: record.bytes,
                    sha256: record.sha256,
                },
            );
        }

        let manifest = NativeAssetManifest {
            schema_version: ASSET_SCHEMA_VERSION,
            generator: GeneratorRecord {
                name: env!("CARGO_PKG_NAME"),
                version: env!("CARGO_PKG_VERSION"),
            },
            sources,
            assets,
            counts: CountRecord {
                forms,
                sandhi_keys,
                sandhi_variants,
                word2vec_vocab: scorer.vocab_size.into(),
                sentencepiece_vocab: scorer.sp_vocab_size.into(),
            },
            scorer: ScorerRecord {
                vector_dimension: scorer.dimension,
                window: scorer.window,
                cbow_mean: scorer.cbow_mean,
                log_table_entries: scorer.log_table_len,
            },
        };
        let mut json = serde_json::to_vec_pretty(&manifest)?;
        json.push(b'\n');
        fs::write(staging.join(MANIFEST_FILE), json)
            .context("failed to write native asset manifest")?;
        Ok(manifest)
    }
}

fn install_asset(staging: &Path, output: &Path, name: &str) -> Result<()> {
    // Move the old entry into staging before installing the replacement. Open
    // handles continue to see the complete old file, and the first move can be
    // rolled back if the second fails. A concurrent open during the brief gap
    // fails loudly; the manifest is installed last and every digest is checked.
    let source = staging.join(name);
    let destination = output.join(name);
    let previous = staging.join(format!(".previous-{name}"));
    let replaced = match fs::rename(&destination, &previous) {
        Ok(()) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to move the previous generated asset {name}"))
        }
    };

    if let Err(error) = fs::rename(&source, &destination) {
        if replaced {
            fs::rename(&previous, &destination).with_context(|| {
                format!("failed to restore the previous generated asset {name} after: {error}")
            })?;
        }
        return Err(error).with_context(|| format!("failed to install generated asset {name}"));
    }
    if replaced {
        fs::remove_file(&previous)
            .with_context(|| format!("failed to remove the previous generated asset {name}"))?;
    }
    Ok(())
}

fn build_forms(input: &Path, output: &Path) -> Result<u64> {
    let reader = BufReader::new(
        File::open(input).with_context(|| format!("failed to open {}", input.display()))?,
    );
    let writer = BufWriter::new(
        File::create(output).with_context(|| format!("failed to create {}", output.display()))?,
    );
    let mut builder = SetBuilder::new(writer)?;
    let mut previous: Option<String> = None;
    let mut count = 0_u64;
    for (line_number, line) in reader.lines().enumerate() {
        let form = line.with_context(|| {
            format!(
                "invalid UTF-8 in {} at line {}",
                input.display(),
                line_number + 1
            )
        })?;
        ensure!(!form.is_empty(), "empty form at line {}", line_number + 1);
        ensure!(
            form.is_ascii(),
            "forms must contain canonical ASCII SLP1: non-ASCII form at line {}",
            line_number + 1
        );
        if let Some(prior) = &previous {
            ensure!(
                prior < &form,
                "forms must be strictly UTF-8 sorted and unique: {:?} then {:?}",
                prior,
                form
            );
        }
        builder.insert(&form)?;
        previous = Some(form);
        count += 1;
    }
    ensure!(count > 0, "forms input is empty");
    builder.finish()?;
    Ok(count)
}

fn build_sandhi(input: &Path, map_output: &Path, variants_output: &Path) -> Result<(u64, u64)> {
    let reader = BufReader::new(
        File::open(input).with_context(|| format!("failed to open {}", input.display()))?,
    );
    let mut groups = Vec::new();
    let mut previous_after: Option<String> = None;
    for (line_number, line) in reader.lines().enumerate() {
        let line = line?;
        ensure!(
            !line.is_empty(),
            "empty sandhi JSONL line {}",
            line_number + 1
        );
        let group: RuleGroup = serde_json::from_str(&line)
            .with_context(|| format!("invalid sandhi JSONL line {}", line_number + 1))?;
        ensure!(
            !group.after.is_empty(),
            "empty sandhi key at line {}",
            line_number + 1
        );
        ensure!(
            group.after.is_ascii(),
            "sandhi keys must contain canonical ASCII SLP1 at line {}",
            line_number + 1
        );
        ensure!(
            !group.variants.is_empty(),
            "sandhi key {:?} has no variants",
            group.after
        );
        for variant in &group.variants {
            ensure!(
                !variant.left.is_empty(),
                "sandhi variant left sides must be non-empty for key {:?}",
                group.after
            );
            ensure!(
                variant.left.is_ascii() && variant.right.is_ascii(),
                "sandhi variant sides must contain canonical ASCII SLP1 for key {:?}",
                group.after
            );
        }
        if let Some(prior) = &previous_after {
            ensure!(
                prior < &group.after,
                "sandhi keys must be strictly sorted and unique"
            );
        }
        ensure!(
            group.variants.windows(2).all(|pair| pair[0] < pair[1]),
            "variants for {:?} must be strictly sorted and unique",
            group.after
        );
        previous_after = Some(group.after.clone());
        groups.push(group);
    }
    ensure!(!groups.is_empty(), "sandhi rules input is empty");

    let map_writer = BufWriter::new(File::create(map_output)?);
    let mut map_builder = MapBuilder::new(map_writer)?;
    for (group_id, group) in groups.iter().enumerate() {
        map_builder.insert(&group.after, group_id as u64)?;
    }
    map_builder.finish()?;

    let header_len = 8 + 4 + 4 + 8 + (groups.len() + 1) * 8;
    let mut payload = Vec::new();
    let mut offsets = Vec::with_capacity(groups.len() + 1);
    let mut variant_count = 0_u64;
    for group in &groups {
        offsets.push((header_len + payload.len()) as u64);
        for variant in &group.variants {
            write_len_prefixed_pair(&mut payload, &variant.left, &variant.right)?;
            variant_count += 1;
        }
    }
    offsets.push((header_len + payload.len()) as u64);

    let group_count = u32::try_from(groups.len()).context("too many sandhi groups")?;
    let mut writer = BufWriter::new(File::create(variants_output)?);
    writer.write_all(VARIANTS_MAGIC)?;
    writer.write_all(&ASSET_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&group_count.to_le_bytes())?;
    writer.write_all(&variant_count.to_le_bytes())?;
    for offset in offsets {
        writer.write_all(&offset.to_le_bytes())?;
    }
    writer.write_all(&payload)?;
    writer.flush()?;
    Ok((groups.len() as u64, variant_count))
}

fn write_len_prefixed_pair(buffer: &mut Vec<u8>, left: &str, right: &str) -> Result<()> {
    let left_len = u32::try_from(left.len()).context("sandhi left value is too long")?;
    let right_len = u32::try_from(right.len()).context("sandhi right value is too long")?;
    buffer.extend_from_slice(&left_len.to_le_bytes());
    buffer.extend_from_slice(&right_len.to_le_bytes());
    buffer.extend_from_slice(left.as_bytes());
    buffer.extend_from_slice(right.as_bytes());
    Ok(())
}

struct ScorerSummary {
    vocab_size: u32,
    sp_vocab_size: u32,
    dimension: u32,
    window: u32,
    cbow_mean: bool,
    log_table_len: u32,
}

fn build_scorer(inputs: &InputPaths, output: &Path, vocab_output: &Path) -> Result<ScorerSummary> {
    let metadata: ScorerMetadata =
        serde_json::from_reader(BufReader::new(File::open(&inputs.scorer_metadata)?))
            .context("invalid scorer metadata")?;
    ensure!(
        metadata.schema_version == 1,
        "unsupported scorer metadata schema"
    );
    ensure!(!metadata.vocab.is_empty(), "scorer vocabulary is empty");
    ensure!(
        metadata.window > 0,
        "scorer window must be greater than zero"
    );

    let syn0: Array2<f32> = read_npy(&inputs.syn0)?;
    let syn1: Array2<f32> = read_npy(&inputs.syn1)?;
    ensure!(syn0.dim() == syn1.dim(), "syn0 and syn1 shapes differ");
    ensure!(
        syn0.nrows() == metadata.vocab.len(),
        "vocabulary and matrix rows differ"
    );
    ensure!(syn0.ncols() > 0, "scorer vector dimension is zero");
    validate_finite("syn0", syn0.iter())?;
    validate_finite("syn1", syn1.iter())?;

    let code_offsets: Array1<u64> = read_npy(&inputs.code_offsets)?;
    let code_data: Array1<u8> = read_npy(&inputs.code_data)?;
    let point_offsets: Array1<u64> = read_npy(&inputs.point_offsets)?;
    let point_data: Array1<u32> = read_npy(&inputs.point_data)?;
    let log_table: Array1<f32> = read_npy(&inputs.log_table)?;
    validate_offsets("code", &code_offsets, code_data.len(), metadata.vocab.len())?;
    validate_offsets(
        "point",
        &point_offsets,
        point_data.len(),
        metadata.vocab.len(),
    )?;
    validate_huffman_lengths(&code_offsets, &point_offsets)?;
    ensure!(
        code_data.iter().all(|value| *value <= 1),
        "Huffman codes must be 0 or 1"
    );
    ensure!(
        point_data
            .iter()
            .all(|value| (*value as usize) < syn1.nrows()),
        "Huffman point references a missing syn1 row"
    );
    ensure!(
        log_table.len() == 1000,
        "log-sigmoid table must contain 1000 entries"
    );
    validate_finite("log-sigmoid table", log_table.iter())?;

    let mut vocab_index = BTreeMap::new();
    for (index, word) in metadata.vocab.iter().enumerate() {
        if vocab_index.insert(word.as_str(), index as i32).is_some() {
            bail!("duplicate scorer vocabulary item {word:?}");
        }
    }
    let vocab_writer = BufWriter::new(
        File::create(vocab_output)
            .with_context(|| format!("failed to create {}", vocab_output.display()))?,
    );
    let mut vocab_builder = MapBuilder::new(vocab_writer)?;
    for (&word, &row) in &vocab_index {
        vocab_builder.insert(word, row as u64)?;
    }
    vocab_builder.finish()?;
    let model = read_sentencepiece_model(&inputs.sentencepiece_model)?;
    ensure!(
        !model.pieces.is_empty(),
        "SentencePiece model contains no pieces"
    );
    let sp_to_vocab: Vec<i32> = model
        .pieces
        .iter()
        .map(|piece| vocab_index.get(piece.piece.as_str()).copied().unwrap_or(-1))
        .collect();

    let vocab_size = u32::try_from(metadata.vocab.len())?;
    let sp_vocab_size = u32::try_from(sp_to_vocab.len())?;
    let dimension = u32::try_from(syn0.ncols())?;
    let log_table_len = u32::try_from(log_table.len())?;
    let mut bytes = vec![0_u8; SCORER_HEADER_LEN];
    let syn0_offset = append_aligned_f32(&mut bytes, contiguous2("syn0", &syn0)?)?;
    let syn1_offset = append_aligned_f32(&mut bytes, contiguous2("syn1", &syn1)?)?;
    let code_offsets_offset =
        append_aligned_u64(&mut bytes, contiguous1("code_offsets", &code_offsets)?)?;
    let code_data_offset = append_aligned_u8(&mut bytes, contiguous1("code_data", &code_data)?)?;
    let point_offsets_offset =
        append_aligned_u64(&mut bytes, contiguous1("point_offsets", &point_offsets)?)?;
    let point_data_offset =
        append_aligned_u32(&mut bytes, contiguous1("point_data", &point_data)?)?;
    let sp_to_vocab_offset = append_aligned_i32(&mut bytes, &sp_to_vocab)?;
    let log_table_offset = append_aligned_f32(&mut bytes, contiguous1("log_table", &log_table)?)?;
    let file_len = bytes.len() as u64;

    let mut header = Vec::with_capacity(SCORER_HEADER_LEN);
    header.extend_from_slice(SCORER_MAGIC);
    for value in [
        ASSET_SCHEMA_VERSION,
        vocab_size,
        dimension,
        sp_vocab_size,
        metadata.window,
        u32::from(metadata.cbow_mean),
        log_table_len,
        0,
    ] {
        header.extend_from_slice(&value.to_le_bytes());
    }
    for value in [
        code_data.len() as u64,
        point_data.len() as u64,
        syn0_offset,
        syn1_offset,
        code_offsets_offset,
        code_data_offset,
        point_offsets_offset,
        point_data_offset,
        sp_to_vocab_offset,
        log_table_offset,
        file_len,
    ] {
        header.extend_from_slice(&value.to_le_bytes());
    }
    ensure!(
        header.len() == SCORER_HEADER_LEN,
        "internal scorer header size mismatch"
    );
    bytes[..SCORER_HEADER_LEN].copy_from_slice(&header);
    fs::write(output, bytes)?;

    Ok(ScorerSummary {
        vocab_size,
        sp_vocab_size,
        dimension,
        window: metadata.window,
        cbow_mean: metadata.cbow_mean,
        log_table_len,
    })
}

fn validate_offsets(
    name: &str,
    offsets: &Array1<u64>,
    data_len: usize,
    vocab_len: usize,
) -> Result<()> {
    ensure!(
        offsets.len() == vocab_len + 1,
        "{name} offsets must have vocab_size + 1 entries"
    );
    ensure!(
        offsets.first() == Some(&0),
        "{name} offsets must start at zero"
    );
    ensure!(
        offsets
            .windows(2)
            .into_iter()
            .all(|window| window[0] <= window[1]),
        "{name} offsets are not monotonic"
    );
    ensure!(
        offsets.last().copied() == Some(data_len as u64),
        "{name} offsets do not cover their data"
    );
    Ok(())
}

fn validate_huffman_lengths(code_offsets: &Array1<u64>, point_offsets: &Array1<u64>) -> Result<()> {
    for row in 0..code_offsets.len() - 1 {
        let code_len = code_offsets[row + 1] - code_offsets[row];
        let point_len = point_offsets[row + 1] - point_offsets[row];
        ensure!(
            code_len == point_len,
            "Huffman code and point lengths disagree at vocabulary row {row}"
        );
    }
    Ok(())
}

fn validate_finite<'a>(name: &str, values: impl IntoIterator<Item = &'a f32>) -> Result<()> {
    ensure!(
        values.into_iter().all(|value| value.is_finite()),
        "{name} contains a non-finite value"
    );
    Ok(())
}

fn contiguous1<'a, T>(name: &str, array: &'a Array1<T>) -> Result<&'a [T]> {
    array
        .as_slice()
        .ok_or_else(|| anyhow!("{name} array is not contiguous"))
}

fn contiguous2<'a, T>(name: &str, array: &'a Array2<T>) -> Result<&'a [T]> {
    array
        .as_slice()
        .ok_or_else(|| anyhow!("{name} array is not contiguous"))
}

fn align(buffer: &mut Vec<u8>) -> u64 {
    let padding = (SECTION_ALIGNMENT - buffer.len() % SECTION_ALIGNMENT) % SECTION_ALIGNMENT;
    buffer.resize(buffer.len() + padding, 0);
    buffer.len() as u64
}

macro_rules! aligned_numeric_writer {
    ($name:ident, $ty:ty) => {
        fn $name(buffer: &mut Vec<u8>, values: &[$ty]) -> Result<u64> {
            let offset = align(buffer);
            for value in values {
                buffer.extend_from_slice(&value.to_le_bytes());
            }
            Ok(offset)
        }
    };
}

aligned_numeric_writer!(append_aligned_f32, f32);
aligned_numeric_writer!(append_aligned_u64, u64);
aligned_numeric_writer!(append_aligned_u32, u32);
aligned_numeric_writer!(append_aligned_i32, i32);

fn append_aligned_u8(buffer: &mut Vec<u8>, values: &[u8]) -> Result<u64> {
    let offset = align(buffer);
    buffer.extend_from_slice(values);
    Ok(offset)
}

#[derive(Clone, PartialEq, Message)]
struct SentencePieceModel {
    #[prost(message, repeated, tag = "1")]
    pieces: Vec<SentencePiece>,
}

#[derive(Clone, PartialEq, Message)]
struct SentencePiece {
    #[prost(string, tag = "1")]
    piece: String,
}

fn read_sentencepiece_model(path: &Path) -> Result<SentencePieceModel> {
    let mut bytes = Vec::new();
    File::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .read_to_end(&mut bytes)?;
    SentencePieceModel::decode(bytes.as_slice())
        .with_context(|| format!("invalid SentencePiece model {}", path.display()))
}

fn source_paths(inputs: &InputPaths) -> [(&'static str, &Path); 11] {
    [
        ("forms", &inputs.forms),
        ("sandhi_rules", &inputs.sandhi_rules),
        ("scorer_metadata", &inputs.scorer_metadata),
        ("syn0", &inputs.syn0),
        ("syn1", &inputs.syn1),
        ("code_offsets", &inputs.code_offsets),
        ("code_data", &inputs.code_data),
        ("point_offsets", &inputs.point_offsets),
        ("point_data", &inputs.point_data),
        ("log_table", &inputs.log_table),
        ("sentencepiece_model", &inputs.sentencepiece_model),
    ]
}

fn file_record(path: impl AsRef<Path>) -> Result<FileRecord> {
    let path = path.as_ref();
    let mut file =
        File::open(path).with_context(|| format!("failed to hash {}", path.display()))?;
    let mut hasher = Sha256::new();
    let bytes = std::io::copy(&mut file, &mut hasher)?;
    Ok(FileRecord {
        bytes,
        sha256: hex::encode(hasher.finalize()),
    })
}

pub fn default_config_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../resources.toml")
}

#[cfg(test)]
mod tests {
    use super::*;
    use fst::{Map, Set};
    use ndarray::{array, Array1};
    use ndarray_npy::write_npy;
    use std::io::{Seek, SeekFrom};
    use tempfile::TempDir;

    struct Fixture {
        _temp: TempDir,
        inputs: InputPaths,
        first_output: PathBuf,
        second_output: PathBuf,
    }

    impl Fixture {
        fn create() -> Result<Self> {
            let temp = tempfile::tempdir()?;
            let input = temp.path().join("input");
            fs::create_dir(&input)?;
            fs::write(input.join("forms.txt"), "a\nab\nb\n")?;
            fs::write(
                input.join("sandhi-rules.jsonl"),
                concat!(
                    "{\"after\":\"a\",\"variants\":[{\"left\":\"A\",\"right\":\"\"}]}\n",
                    "{\"after\":\"b\",\"variants\":[{\"left\":\"a\",\"right\":\"b\"},{\"left\":\"c\",\"right\":\"d\"}]}\n"
                ),
            )?;
            fs::write(
                input.join("scorer.json"),
                r#"{"schema_version":1,"window":5,"cbow_mean":true,"vocab":["a","b"]}"#,
            )?;

            write_npy(input.join("syn0.npy"), &array![[1.0_f32, 2.0], [3.0, 4.0]])?;
            write_npy(input.join("syn1.npy"), &array![[5.0_f32, 6.0], [7.0, 8.0]])?;
            write_npy(input.join("code-offsets.npy"), &array![0_u64, 1, 3])?;
            write_npy(input.join("code-data.npy"), &array![0_u8, 1, 0])?;
            write_npy(input.join("point-offsets.npy"), &array![0_u64, 1, 3])?;
            write_npy(input.join("point-data.npy"), &array![0_u32, 0, 1])?;
            write_npy(input.join("log-table.npy"), &Array1::<f32>::zeros(1000))?;
            let sentencepiece = SentencePieceModel {
                pieces: vec![
                    SentencePiece { piece: "a".into() },
                    SentencePiece {
                        piece: "<unk>".into(),
                    },
                    SentencePiece { piece: "b".into() },
                ],
            };
            fs::write(
                input.join("sentencepiece.model"),
                sentencepiece.encode_to_vec(),
            )?;

            let inputs = InputPaths {
                forms: input.join("forms.txt"),
                sandhi_rules: input.join("sandhi-rules.jsonl"),
                scorer_metadata: input.join("scorer.json"),
                syn0: input.join("syn0.npy"),
                syn1: input.join("syn1.npy"),
                code_offsets: input.join("code-offsets.npy"),
                code_data: input.join("code-data.npy"),
                point_offsets: input.join("point-offsets.npy"),
                point_data: input.join("point-data.npy"),
                log_table: input.join("log-table.npy"),
                sentencepiece_model: input.join("sentencepiece.model"),
            };
            let first_output = temp.path().join("output-one");
            let second_output = temp.path().join("output-two");
            Ok(Self {
                _temp: temp,
                inputs,
                first_output,
                second_output,
            })
        }

        fn config(&self, output: PathBuf) -> BuilderConfig {
            BuilderConfig {
                schema_version: CONFIG_SCHEMA_VERSION,
                generation: None,
                inputs: self.inputs.clone(),
                output: OutputConfig { directory: output },
            }
        }

        fn expect_build_error(&self, expected: &str) -> Result<()> {
            let error = ResourceBuilder::new(self.config(self.first_output.clone()))
                .build()
                .expect_err("invalid resource input must fail");
            let message = error.to_string();
            ensure!(
                message.contains(expected),
                "expected error containing {expected:?}, got {message:?}"
            );
            ensure!(
                !self.first_output.join(MANIFEST_FILE).exists(),
                "a rejected build must not install its manifest"
            );
            Ok(())
        }
    }

    #[test]
    fn build_is_deterministic_and_matches_the_binary_contract() -> Result<()> {
        let fixture = Fixture::create()?;
        let first = ResourceBuilder::new(fixture.config(fixture.first_output.clone())).build()?;
        ResourceBuilder::new(fixture.config(fixture.second_output.clone())).build()?;
        assert_eq!(first.counts.forms, 3);
        assert_eq!(first.counts.sandhi_keys, 2);
        assert_eq!(first.counts.sandhi_variants, 3);
        assert_eq!(first.counts.word2vec_vocab, 2);
        assert_eq!(first.counts.sentencepiece_vocab, 3);

        for name in [
            FORMS_FILE,
            SANDHI_MAP_FILE,
            SANDHI_VARIANTS_FILE,
            SCORER_FILE,
            SCORER_VOCAB_FILE,
            SENTENCEPIECE_FILE,
            MANIFEST_FILE,
        ] {
            assert_eq!(
                fs::read(fixture.first_output.join(name))?,
                fs::read(fixture.second_output.join(name))?,
                "{name} changed between equivalent builds"
            );
        }

        let forms = Set::new(fs::read(fixture.first_output.join(FORMS_FILE))?)?;
        assert!(forms.contains("ab"));
        assert!(!forms.contains("c"));
        let rules = Map::new(fs::read(fixture.first_output.join(SANDHI_MAP_FILE))?)?;
        assert_eq!(rules.get("a"), Some(0));
        assert_eq!(rules.get("b"), Some(1));

        let variants = fs::read(fixture.first_output.join(SANDHI_VARIANTS_FILE))?;
        assert_eq!(&variants[..8], VARIANTS_MAGIC);
        assert_eq!(u32::from_le_bytes(variants[8..12].try_into()?), 1);
        assert_eq!(u32::from_le_bytes(variants[12..16].try_into()?), 2);
        assert_eq!(u64::from_le_bytes(variants[16..24].try_into()?), 3);

        let scorer = fs::read(fixture.first_output.join(SCORER_FILE))?;
        assert_eq!(&scorer[..8], SCORER_MAGIC);
        assert_eq!(u32::from_le_bytes(scorer[12..16].try_into()?), 2);
        assert_eq!(u32::from_le_bytes(scorer[16..20].try_into()?), 2);
        assert_eq!(u32::from_le_bytes(scorer[20..24].try_into()?), 3);
        let mapping_offset = u64::from_le_bytes(scorer[104..112].try_into()?) as usize;
        let mapping: Vec<i32> = scorer[mapping_offset..mapping_offset + 12]
            .chunks_exact(4)
            .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("four bytes")))
            .collect();
        assert_eq!(mapping, vec![0, -1, 1]);
        let scorer_vocab = Map::new(fs::read(fixture.first_output.join(SCORER_VOCAB_FILE))?)?;
        assert_eq!(scorer_vocab.get("a"), Some(0));
        assert_eq!(scorer_vocab.get("b"), Some(1));
        Ok(())
    }

    #[test]
    fn rejects_noncanonical_source_order() -> Result<()> {
        let fixture = Fixture::create()?;
        fs::write(&fixture.inputs.forms, "b\na\n")?;
        let error = ResourceBuilder::new(fixture.config(fixture.first_output.clone()))
            .build()
            .expect_err("unsorted forms must fail");
        assert!(error.to_string().contains("strictly UTF-8 sorted"));
        assert!(!fixture.first_output.join(MANIFEST_FILE).exists());
        Ok(())
    }

    #[test]
    fn rejects_non_ascii_forms() -> Result<()> {
        let fixture = Fixture::create()?;
        fs::write(&fixture.inputs.forms, "a\nā\n")?;
        fixture.expect_build_error("forms must contain canonical ASCII SLP1")
    }

    #[test]
    fn rejects_invalid_sandhi_text() -> Result<()> {
        let cases = [
            (
                "non-ASCII after",
                r#"{"after":"ā","variants":[{"left":"a","right":""}]}"#,
                "sandhi keys must contain canonical ASCII SLP1",
            ),
            (
                "empty left",
                r#"{"after":"a","variants":[{"left":"","right":"b"}]}"#,
                "sandhi variant left sides must be non-empty",
            ),
            (
                "non-ASCII left",
                r#"{"after":"a","variants":[{"left":"ā","right":"b"}]}"#,
                "sandhi variant sides must contain canonical ASCII SLP1",
            ),
            (
                "non-ASCII right",
                r#"{"after":"a","variants":[{"left":"a","right":"ā"}]}"#,
                "sandhi variant sides must contain canonical ASCII SLP1",
            ),
        ];
        for (label, json, expected) in cases {
            let fixture = Fixture::create()?;
            fs::write(&fixture.inputs.sandhi_rules, format!("{json}\n"))?;
            fixture
                .expect_build_error(expected)
                .with_context(|| format!("case {label}"))?;
        }
        Ok(())
    }

    #[test]
    fn rejects_zero_scorer_window() -> Result<()> {
        let fixture = Fixture::create()?;
        fs::write(
            &fixture.inputs.scorer_metadata,
            r#"{"schema_version":1,"window":0,"cbow_mean":true,"vocab":["a","b"]}"#,
        )?;

        fixture.expect_build_error("scorer window must be greater than zero")
    }

    #[test]
    fn rejects_non_finite_scorer_values() -> Result<()> {
        let fixture = Fixture::create()?;
        write_npy(&fixture.inputs.syn0, &array![[f32::NAN, 2.0], [3.0, 4.0]])?;
        fixture.expect_build_error("syn0 contains a non-finite value")?;

        let fixture = Fixture::create()?;
        write_npy(
            &fixture.inputs.syn1,
            &array![[5.0_f32, f32::INFINITY], [7.0, 8.0]],
        )?;
        fixture.expect_build_error("syn1 contains a non-finite value")?;

        let fixture = Fixture::create()?;
        let mut log_table = Array1::<f32>::zeros(1000);
        log_table[999] = f32::NEG_INFINITY;
        write_npy(&fixture.inputs.log_table, &log_table)?;
        fixture.expect_build_error("log-sigmoid table contains a non-finite value")
    }

    #[test]
    fn rejects_per_row_huffman_length_mismatch() -> Result<()> {
        let fixture = Fixture::create()?;
        write_npy(&fixture.inputs.point_offsets, &array![0_u64, 2, 3])?;

        fixture.expect_build_error("Huffman code and point lengths disagree at vocabulary row 0")
    }

    #[test]
    fn repeated_build_replaces_assets_without_truncating_open_files() -> Result<()> {
        let fixture = Fixture::create()?;
        let config = fixture.config(fixture.first_output.clone());
        ResourceBuilder::new(config.clone()).build()?;

        let forms_path = fixture.first_output.join(FORMS_FILE);
        let original = fs::read(&forms_path)?;
        let mut opened_before_rebuild = File::open(&forms_path)?;

        fs::write(&fixture.inputs.forms, "a\nab\nb\nc\n")?;
        ResourceBuilder::new(config).build()?;
        let replacement = fs::read(&forms_path)?;

        assert_ne!(
            replacement, original,
            "the rebuilt forms asset did not change"
        );
        let mut still_open = Vec::new();
        opened_before_rebuild.seek(SeekFrom::Start(0))?;
        opened_before_rebuild.read_to_end(&mut still_open)?;
        assert_eq!(
            still_open, original,
            "rebuilding truncated or modified an already-open asset"
        );
        assert!(fixture.first_output.join(MANIFEST_FILE).exists());
        Ok(())
    }

    #[test]
    fn config_paths_are_resolved_from_the_config_file() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let config_dir = temp.path().join("config");
        fs::create_dir(&config_dir)?;
        let path = config_dir.join("resources.toml");
        fs::write(
            &path,
            r#"
schema_version = 1

[generation]
legacy_data = "legacy"
neutral_output = "neutral"

[inputs]
forms = "input/forms.txt"
sandhi_rules = "input/rules.jsonl"
scorer_metadata = "input/scorer.json"
syn0 = "input/syn0.npy"
syn1 = "input/syn1.npy"
code_offsets = "input/code-offsets.npy"
code_data = "input/code-data.npy"
point_offsets = "input/point-offsets.npy"
point_data = "input/point-data.npy"
log_table = "input/log-table.npy"
sentencepiece_model = "input/sentencepiece.model"

[output]
directory = "native"
"#,
        )?;

        let config = BuilderConfig::from_path(&path)?;
        assert_eq!(config.inputs.forms, config_dir.join("input/forms.txt"));
        assert_eq!(config.output.directory, config_dir.join("native"));
        let generation = config.generation.expect("generation config");
        assert_eq!(generation.legacy_data, config_dir.join("legacy"));
        assert_eq!(generation.neutral_output, config_dir.join("neutral"));
        Ok(())
    }
}
