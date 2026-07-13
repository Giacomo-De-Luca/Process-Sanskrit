use std::fs;
use std::path::Path;
use std::sync::Arc;

use fst::Map;

use crate::CoreError;

const SCORER_MAGIC: &[u8; 8] = b"PSSC0001";
const SCORER_VERSION: u32 = 1;
const HEADER_LEN: usize = 128;
const MAX_EXP: f32 = 6.0;
const INDEX_SCALE: f32 = 83.0;

/// Native SentencePiece adapter. The Python binding supplies an implementation
/// backed by the statically linked official SentencePiece library.
pub trait PieceEncoder: Send + Sync + std::fmt::Debug {
    fn encode(&self, text: &str) -> Result<Vec<String>, CoreError>;
}

/// Scores complete SLP1 token sequences.
pub trait SequenceScorer: Send + Sync {
    fn score_sequence(&self, tokens: &[&str]) -> Result<f32, CoreError>;

    fn score_sequences(&self, sequences: &[Vec<&str>]) -> Result<Vec<f32>, CoreError> {
        sequences
            .iter()
            .map(|sequence| self.score_sequence(sequence))
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
struct Section {
    offset: usize,
    len: usize,
}

/// Validated, process-owned DCS CBOW/hierarchical-softmax weights.
pub struct ScorerModel {
    vocab: Map<Vec<u8>>,
    vocab_size: usize,
    dim: usize,
    sp_vocab_size: usize,
    window: usize,
    cbow_mean: bool,
    syn0: Vec<f32>,
    syn1: Vec<f32>,
    code_offsets: Vec<u64>,
    code_data: Vec<u8>,
    point_offsets: Vec<u64>,
    point_data: Vec<u32>,
    sp_to_vocab: Vec<i32>,
    log_table: Vec<f32>,
}

impl std::fmt::Debug for ScorerModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ScorerModel")
            .field("vocab_size", &self.vocab_size)
            .field("dim", &self.dim)
            .field("sp_vocab_size", &self.sp_vocab_size)
            .field("window", &self.window)
            .field("cbow_mean", &self.cbow_mean)
            .finish_non_exhaustive()
    }
}

impl ScorerModel {
    pub fn load(path: &Path, vocab_path: &Path) -> Result<Self, CoreError> {
        let bytes = fs::read(path).map_err(|error| CoreError::io(path, error))?;
        let vocab = fs::read(vocab_path).map_err(|error| CoreError::io(vocab_path, error))?;
        Self::from_owned_bytes(bytes, vocab)
    }

    pub(crate) fn from_owned_bytes(bytes: Vec<u8>, vocab: Vec<u8>) -> Result<Self, CoreError> {
        if !cfg!(target_endian = "little") {
            return Err(CoreError::asset(
                "scorer.bin",
                "only little-endian targets are supported",
            ));
        }
        let vocab = Map::new(vocab)
            .map_err(|error| CoreError::asset("scorer_vocab.fst", error.to_string()))?;
        Self::parse(bytes, vocab)
    }

    fn parse(bytes: Vec<u8>, vocab: Map<Vec<u8>>) -> Result<Self, CoreError> {
        if bytes.len() < HEADER_LEN || &bytes[..8] != SCORER_MAGIC {
            return Err(CoreError::asset("scorer.bin", "missing PSSC0001 header"));
        }
        let mut cursor = 8;
        let version = header_u32(&bytes, &mut cursor)?;
        if version != SCORER_VERSION {
            return Err(CoreError::asset(
                "scorer.bin",
                format!("unsupported format version {version}"),
            ));
        }
        let vocab_size = header_u32(&bytes, &mut cursor)? as usize;
        let dim = header_u32(&bytes, &mut cursor)? as usize;
        let sp_vocab_size = header_u32(&bytes, &mut cursor)? as usize;
        let window = header_u32(&bytes, &mut cursor)? as usize;
        let cbow_mean_raw = header_u32(&bytes, &mut cursor)?;
        let log_table_len = header_u32(&bytes, &mut cursor)? as usize;
        let reserved = header_u32(&bytes, &mut cursor)?;
        if vocab_size == 0 || dim == 0 || sp_vocab_size == 0 || window == 0 {
            return Err(CoreError::asset("scorer.bin", "zero-sized model dimension"));
        }
        if cbow_mean_raw > 1 || reserved != 0 || log_table_len != 1000 {
            return Err(CoreError::asset(
                "scorer.bin",
                "invalid CBOW flag, reserved field, or sigmoid table length",
            ));
        }

        let code_count = usize_from_u64(header_u64(&bytes, &mut cursor)?)?;
        let point_count = usize_from_u64(header_u64(&bytes, &mut cursor)?)?;
        let syn0_off = usize_from_u64(header_u64(&bytes, &mut cursor)?)?;
        let syn1_off = usize_from_u64(header_u64(&bytes, &mut cursor)?)?;
        let code_offsets_off = usize_from_u64(header_u64(&bytes, &mut cursor)?)?;
        let code_data_off = usize_from_u64(header_u64(&bytes, &mut cursor)?)?;
        let point_offsets_off = usize_from_u64(header_u64(&bytes, &mut cursor)?)?;
        let point_data_off = usize_from_u64(header_u64(&bytes, &mut cursor)?)?;
        let sp_to_vocab_off = usize_from_u64(header_u64(&bytes, &mut cursor)?)?;
        let log_table_off = usize_from_u64(header_u64(&bytes, &mut cursor)?)?;
        let file_len = usize_from_u64(header_u64(&bytes, &mut cursor)?)?;
        if cursor != HEADER_LEN || file_len != bytes.len() {
            return Err(CoreError::asset(
                "scorer.bin",
                "header length or declared file length is invalid",
            ));
        }

        let matrix_items = vocab_size
            .checked_mul(dim)
            .ok_or_else(|| CoreError::asset("scorer.bin", "matrix length overflow"))?;
        let syn0 = section(syn0_off, matrix_items, 4, syn1_off, bytes.len(), "syn0")?;
        let syn1 = section(
            syn1_off,
            matrix_items,
            4,
            code_offsets_off,
            bytes.len(),
            "syn1",
        )?;
        let code_offsets = section(
            code_offsets_off,
            vocab_size + 1,
            8,
            code_data_off,
            bytes.len(),
            "code offsets",
        )?;
        let code_data = section(
            code_data_off,
            code_count,
            1,
            point_offsets_off,
            bytes.len(),
            "code data",
        )?;
        let point_offsets = section(
            point_offsets_off,
            vocab_size + 1,
            8,
            point_data_off,
            bytes.len(),
            "point offsets",
        )?;
        let point_data = section(
            point_data_off,
            point_count,
            4,
            sp_to_vocab_off,
            bytes.len(),
            "point data",
        )?;
        let sp_to_vocab = section(
            sp_to_vocab_off,
            sp_vocab_size,
            4,
            log_table_off,
            bytes.len(),
            "SentencePiece map",
        )?;
        let log_table = section(
            log_table_off,
            log_table_len,
            4,
            bytes.len(),
            bytes.len(),
            "sigmoid table",
        )?;

        let model = Self {
            vocab,
            vocab_size,
            dim,
            sp_vocab_size,
            window,
            cbow_mean: cbow_mean_raw == 1,
            syn0: decode_section(&bytes, syn0, "syn0")?,
            syn1: decode_section(&bytes, syn1, "syn1")?,
            code_offsets: decode_section(&bytes, code_offsets, "code offsets")?,
            code_data: decode_section(&bytes, code_data, "code data")?,
            point_offsets: decode_section(&bytes, point_offsets, "point offsets")?,
            point_data: decode_section(&bytes, point_data, "point data")?,
            sp_to_vocab: decode_section(&bytes, sp_to_vocab, "SentencePiece map")?,
            log_table: decode_section(&bytes, log_table, "sigmoid table")?,
        };
        model.validate_indices(code_count, point_count)?;
        Ok(model)
    }

    fn validate_indices(&self, code_count: usize, point_count: usize) -> Result<(), CoreError> {
        if self.vocab.len() != self.vocab_size
            || self
                .vocab
                .stream()
                .into_values()
                .into_iter()
                .any(|row| row as usize >= self.vocab_size)
        {
            return Err(CoreError::asset(
                "scorer_vocab.fst",
                "vocabulary size or row mapping is inconsistent with scorer.bin",
            ));
        }
        validate_offsets(self.code_offsets(), code_count, "code")?;
        validate_offsets(self.point_offsets(), point_count, "point")?;
        if self
            .code_offsets()
            .windows(2)
            .zip(self.point_offsets().windows(2))
            .any(|(code, point)| code[1] - code[0] != point[1] - point[0])
        {
            return Err(CoreError::asset(
                "scorer.bin",
                "Huffman code and point lengths disagree",
            ));
        }
        if self.code_data().iter().any(|code| *code > 1) {
            return Err(CoreError::asset(
                "scorer.bin",
                "Huffman code data contains a value other than zero or one",
            ));
        }
        if self
            .point_data()
            .iter()
            .any(|point| *point as usize >= self.vocab_size)
        {
            return Err(CoreError::asset(
                "scorer.bin",
                "Huffman point data refers past syn1",
            ));
        }
        if self
            .sp_to_vocab()
            .iter()
            .any(|row| *row < -1 || (*row >= 0 && *row as usize >= self.vocab_size))
        {
            return Err(CoreError::asset(
                "scorer.bin",
                "SentencePiece-to-word2vec mapping is out of range",
            ));
        }
        if self
            .syn0()
            .iter()
            .chain(self.syn1())
            .chain(self.log_table())
            .any(|value| !value.is_finite())
        {
            return Err(CoreError::asset(
                "scorer.bin",
                "model weights or sigmoid table contain a non-finite value",
            ));
        }
        Ok(())
    }

    fn score_pieces(&self, pieces: &[String]) -> Result<f32, CoreError> {
        // Python scores EncodeAsPieces() strings, not numeric IDs. That detail
        // matters for unknown spans: SentencePiece ID 0 still carries the
        // runtime surface (for example `Q`), which may exist in the DCS vocab.
        let words = pieces
            .iter()
            .filter_map(|piece| self.vocab.get(piece).map(|row| row as usize))
            .collect::<Vec<_>>();

        let syn0 = self.syn0();
        let syn1 = self.syn1();
        let code_offsets = self.code_offsets();
        let codes = self.code_data();
        let point_offsets = self.point_offsets();
        let points = self.point_data();
        let log_table = self.log_table();
        let mut total = 0.0_f32;
        let mut neu1 = vec![0.0_f32; self.dim];

        for position in 0..words.len() {
            neu1.fill(0.0);
            let lo = position.saturating_sub(self.window);
            let hi = (position + self.window + 1).min(words.len());
            let mut context_count = 0_usize;
            for (context_position, &context_word) in words.iter().enumerate().take(hi).skip(lo) {
                if context_position == position {
                    continue;
                }
                let row = context_word * self.dim;
                for dimension in 0..self.dim {
                    neu1[dimension] += syn0[row + dimension];
                }
                context_count += 1;
            }
            if self.cbow_mean && context_count != 0 {
                let scale = 1.0_f32 / context_count as f32;
                for value in &mut neu1 {
                    *value *= scale;
                }
            }

            let word = words[position];
            let code_start = code_offsets[word] as usize;
            let code_end = code_offsets[word + 1] as usize;
            let point_start = point_offsets[word] as usize;
            let point_end = point_offsets[word + 1] as usize;
            if code_end - code_start != point_end - point_start {
                return Err(CoreError::asset(
                    "scorer.bin",
                    format!("code and point lengths disagree for vocabulary row {word}"),
                ));
            }
            let mut word_total = 0.0_f32;
            for offset in 0..(code_end - code_start) {
                let point = points[point_start + offset] as usize;
                let row = point * self.dim;
                let mut activation = 0.0_f32;
                for dimension in 0..self.dim {
                    activation += neu1[dimension] * syn1[row + dimension];
                }
                if codes[code_start + offset] == 1 {
                    activation = -activation;
                }
                // gensim deliberately skips saturated terms instead of adding
                // their near-zero log probability.
                if activation <= -MAX_EXP || activation >= MAX_EXP {
                    continue;
                }
                let table_index = ((activation + MAX_EXP) * INDEX_SCALE) as usize;
                word_total += log_table[table_index];
            }
            total += word_total;
        }
        Ok(total)
    }

    fn syn0(&self) -> &[f32] {
        &self.syn0
    }

    fn syn1(&self) -> &[f32] {
        &self.syn1
    }

    fn code_offsets(&self) -> &[u64] {
        &self.code_offsets
    }

    fn code_data(&self) -> &[u8] {
        &self.code_data
    }

    fn point_offsets(&self) -> &[u64] {
        &self.point_offsets
    }

    fn point_data(&self) -> &[u32] {
        &self.point_data
    }

    fn sp_to_vocab(&self) -> &[i32] {
        &self.sp_to_vocab
    }

    fn log_table(&self) -> &[f32] {
        &self.log_table
    }
}

/// Exact DCS sequence scorer using externally supplied SentencePiece surfaces.
#[derive(Debug)]
pub struct DcsScorer {
    model: ScorerModel,
    encoder: Arc<dyn PieceEncoder>,
}

impl DcsScorer {
    pub fn new(model: ScorerModel, encoder: Arc<dyn PieceEncoder>) -> Self {
        Self { model, encoder }
    }

    pub fn score_text(&self, text: &str) -> Result<f32, CoreError> {
        let pieces = self.encoder.encode(text)?;
        self.model.score_pieces(&pieces)
    }
}

impl SequenceScorer for DcsScorer {
    fn score_sequence(&self, tokens: &[&str]) -> Result<f32, CoreError> {
        self.score_text(&tokens.join(" "))
    }
}

trait LittleEndianValue: Sized {
    const WIDTH: usize;

    fn decode(bytes: &[u8]) -> Self;
}

macro_rules! little_endian_value {
    ($value:ty, $width:expr) => {
        impl LittleEndianValue for $value {
            const WIDTH: usize = $width;

            fn decode(bytes: &[u8]) -> Self {
                let raw: [u8; $width] = bytes.try_into().expect("chunk width is fixed");
                Self::from_le_bytes(raw)
            }
        }
    };
}

little_endian_value!(f32, 4);
little_endian_value!(u64, 8);
little_endian_value!(u32, 4);
little_endian_value!(i32, 4);

impl LittleEndianValue for u8 {
    const WIDTH: usize = 1;

    fn decode(bytes: &[u8]) -> Self {
        bytes[0]
    }
}

fn decode_section<T: LittleEndianValue>(
    bytes: &[u8],
    section: Section,
    name: &str,
) -> Result<Vec<T>, CoreError> {
    let byte_len = section
        .len
        .checked_mul(T::WIDTH)
        .ok_or_else(|| CoreError::asset("scorer.bin", format!("{name} length overflow")))?;
    let end = section
        .offset
        .checked_add(byte_len)
        .ok_or_else(|| CoreError::asset("scorer.bin", format!("{name} offset overflow")))?;
    let encoded = bytes.get(section.offset..end).ok_or_else(|| {
        CoreError::asset("scorer.bin", format!("{name} extends past end of file"))
    })?;
    Ok(encoded.chunks_exact(T::WIDTH).map(T::decode).collect())
}

fn header_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, CoreError> {
    let next = *cursor + 4;
    let raw: [u8; 4] = bytes
        .get(*cursor..next)
        .ok_or_else(|| CoreError::asset("scorer.bin", "truncated header"))?
        .try_into()
        .expect("slice length is checked");
    *cursor = next;
    Ok(u32::from_le_bytes(raw))
}

fn header_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, CoreError> {
    let next = *cursor + 8;
    let raw: [u8; 8] = bytes
        .get(*cursor..next)
        .ok_or_else(|| CoreError::asset("scorer.bin", "truncated header"))?
        .try_into()
        .expect("slice length is checked");
    *cursor = next;
    Ok(u64::from_le_bytes(raw))
}

fn usize_from_u64(value: u64) -> Result<usize, CoreError> {
    usize::try_from(value)
        .map_err(|_| CoreError::asset("scorer.bin", "offset exceeds address space"))
}

fn section(
    offset: usize,
    items: usize,
    item_size: usize,
    next_offset: usize,
    file_len: usize,
    name: &str,
) -> Result<Section, CoreError> {
    if offset < HEADER_LEN || offset % 64 != 0 {
        return Err(CoreError::asset(
            "scorer.bin",
            format!("{name} is not 64-byte aligned"),
        ));
    }
    let bytes = items
        .checked_mul(item_size)
        .ok_or_else(|| CoreError::asset("scorer.bin", format!("{name} length overflow")))?;
    let end = offset
        .checked_add(bytes)
        .ok_or_else(|| CoreError::asset("scorer.bin", format!("{name} offset overflow")))?;
    if end > next_offset || next_offset > file_len {
        return Err(CoreError::asset(
            "scorer.bin",
            format!("{name} overlaps the following section or end of file"),
        ));
    }
    Ok(Section { offset, len: items })
}

fn validate_offsets(offsets: &[u64], data_len: usize, name: &str) -> Result<(), CoreError> {
    if offsets.first().copied() != Some(0)
        || offsets.last().copied() != Some(data_len as u64)
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(CoreError::asset(
            "scorer.bin",
            format!("{name} offsets are invalid or non-monotonic"),
        ));
    }
    Ok(())
}
