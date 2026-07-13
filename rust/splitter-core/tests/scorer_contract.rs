use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use fst::MapBuilder;
use process_sanskrit_splitter_core::{CoreError, DcsScorer, PieceEncoder, ScorerModel};

#[derive(Debug)]
struct FixedEncoder(Vec<String>);

impl PieceEncoder for FixedEncoder {
    fn encode(&self, _text: &str) -> Result<Vec<String>, CoreError> {
        Ok(self.0.clone())
    }
}

#[test]
fn runtime_unknown_surfaces_use_the_word2vec_vocabulary() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("scorer.bin");
    let vocab = temp.path().join("scorer_vocab.fst");
    write_tiny_scorer(&path, 1.0, 1.0);
    write_tiny_vocab(&vocab, &[("Q", 0)]);
    let model = ScorerModel::load(&path, &vocab).unwrap();
    let scorer = DcsScorer::new(model, Arc::new(FixedEncoder(vec!["Q".into()])));

    assert_eq!(scorer.score_text("ignored").unwrap(), -0.75);
}

#[test]
fn pieces_missing_from_word2vec_are_dropped_instead_of_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("scorer.bin");
    let vocab = temp.path().join("scorer_vocab.fst");
    write_tiny_scorer(&path, 1.0, 1.0);
    write_tiny_vocab(&vocab, &[("known", 0)]);
    let model = ScorerModel::load(&path, &vocab).unwrap();
    let scorer = DcsScorer::new(model, Arc::new(FixedEncoder(vec!["not-in-vocab".into()])));

    assert_eq!(scorer.score_text("ignored").unwrap(), 0.0);
}

#[test]
fn saturated_huffman_terms_contribute_exactly_zero() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("scorer.bin");
    let vocab = temp.path().join("scorer_vocab.fst");
    write_tiny_scorer(&path, 7.0, 1.0);
    write_tiny_vocab(&vocab, &[("known", 0)]);
    let model = ScorerModel::load(&path, &vocab).unwrap();
    let scorer = DcsScorer::new(
        model,
        Arc::new(FixedEncoder(vec!["known".into(), "known".into()])),
    );

    assert_eq!(scorer.score_text("ignored").unwrap(), 0.0);
}

#[test]
fn loaded_scorer_survives_later_package_file_mutation() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("scorer.bin");
    let vocab = temp.path().join("scorer_vocab.fst");
    write_tiny_scorer(&path, 1.0, 1.0);
    write_tiny_vocab(&vocab, &[("Q", 0)]);
    let model = ScorerModel::load(&path, &vocab).unwrap();
    let scorer = DcsScorer::new(model, Arc::new(FixedEncoder(vec!["Q".into()])));

    let scorer_len = fs::metadata(&path).unwrap().len() as usize;
    let vocab_len = fs::metadata(&vocab).unwrap().len() as usize;
    fs::write(&path, vec![0_u8; scorer_len]).unwrap();
    fs::write(&vocab, vec![0_u8; vocab_len]).unwrap();

    assert_eq!(scorer.score_text("ignored").unwrap(), -0.75);
}

#[test]
fn packaged_scorer_matches_the_numpy_reference() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../process_sanskrit/splitter/data/native/scorer.bin");
    let vocab = path.with_file_name("scorer_vocab.fst");
    let model = ScorerModel::load(&path, &vocab).unwrap();
    let scorer = DcsScorer::new(
        model,
        Arc::new(FixedEncoder(vec![
            "▁asti".into(),
            "▁uttara".into(),
            "syAm".into(),
            "▁diSi".into(),
        ])),
    );

    let actual = scorer.score_text("asti uttarasyAm diSi").unwrap();
    let numpy_reference = -26.654_55_f32;
    assert!((actual - numpy_reference).abs() <= 1.0e-3, "{actual}");
}

#[test]
fn packaged_scorer_matches_numpy_for_unknown_surface_letters() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../process_sanskrit/splitter/data/native/scorer.bin");
    let vocab = path.with_file_name("scorer_vocab.fst");
    let model = ScorerModel::load(&path, &vocab).unwrap();
    let scorer = DcsScorer::new(
        model,
        Arc::new(FixedEncoder(vec![
            "▁".into(),
            "J".into(),
            "▁".into(),
            "Q".into(),
        ])),
    );

    let actual = scorer.score_text("J Q").unwrap();
    let numpy_reference = -29.999_74_f32;
    assert!((actual - numpy_reference).abs() <= 1.0e-3, "{actual}");
}

fn write_tiny_vocab(path: &std::path::Path, entries: &[(&str, u64)]) {
    let mut builder = MapBuilder::new(fs::File::create(path).unwrap()).unwrap();
    for &(piece, row) in entries {
        builder.insert(piece, row).unwrap();
    }
    builder.finish().unwrap();
}

fn write_tiny_scorer(path: &std::path::Path, syn0_value: f32, syn1_value: f32) {
    const HEADER_LEN: usize = 128;
    let mut bytes = vec![0_u8; HEADER_LEN];
    let syn0 = append_f32(&mut bytes, &[syn0_value]);
    let syn1 = append_f32(&mut bytes, &[syn1_value]);
    let code_offsets = append_u64(&mut bytes, &[0, 1]);
    let code_data = append_u8(&mut bytes, &[0]);
    let point_offsets = append_u64(&mut bytes, &[0, 1]);
    let point_data = append_u32(&mut bytes, &[0]);
    let sp_to_vocab = append_i32(&mut bytes, &[0, -1]);
    let mut log_table = vec![0.0_f32; 1000];
    log_table[498] = -0.75;
    let log_table = append_f32(&mut bytes, &log_table);
    let file_len = bytes.len() as u64;

    let mut header = Vec::new();
    header.extend_from_slice(b"PSSC0001");
    for value in [1_u32, 1, 1, 2, 5, 1, 1000, 0] {
        header.extend_from_slice(&value.to_le_bytes());
    }
    for value in [
        1_u64,
        1,
        syn0,
        syn1,
        code_offsets,
        code_data,
        point_offsets,
        point_data,
        sp_to_vocab,
        log_table,
        file_len,
    ] {
        header.extend_from_slice(&value.to_le_bytes());
    }
    assert_eq!(header.len(), HEADER_LEN);
    bytes[..HEADER_LEN].copy_from_slice(&header);
    fs::write(path, bytes).unwrap();
}

fn align(bytes: &mut Vec<u8>) -> u64 {
    let padding = (64 - bytes.len() % 64) % 64;
    bytes.resize(bytes.len() + padding, 0);
    bytes.len() as u64
}

macro_rules! numeric_writer {
    ($name:ident, $ty:ty) => {
        fn $name(bytes: &mut Vec<u8>, values: &[$ty]) -> u64 {
            let offset = align(bytes);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            offset
        }
    };
}

numeric_writer!(append_f32, f32);
numeric_writer!(append_u64, u64);
numeric_writer!(append_u32, u32);
numeric_writer!(append_i32, i32);

fn append_u8(bytes: &mut Vec<u8>, values: &[u8]) -> u64 {
    let offset = align(bytes);
    bytes.extend_from_slice(values);
    offset
}
