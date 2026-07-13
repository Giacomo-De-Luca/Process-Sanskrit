use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use process_sanskrit_splitter_core::{
    CoreError, PieceEncoder, Resources, RuleVariant, SandhiRules, SequenceScorer, SplitOptions,
    Splitter,
};

#[derive(Debug)]
struct RejectingEncoder;

impl PieceEncoder for RejectingEncoder {
    fn encode(&self, _text: &str) -> Result<Vec<String>, CoreError> {
        Err(CoreError::ScorerUnavailable(
            "test encoder must not be used".to_owned(),
        ))
    }
}

#[derive(Debug)]
struct RejectingScorer;

impl SequenceScorer for RejectingScorer {
    fn score_sequence(&self, _tokens: &[&str]) -> Result<f32, CoreError> {
        Err(CoreError::ScorerUnavailable(
            "test scorer must fail before limit validation".to_owned(),
        ))
    }
}

fn resources(forms: &[&str], rules: &[(&str, &[(&str, &str)])]) -> Arc<Resources> {
    let forms = forms
        .iter()
        .map(|word| (*word).to_owned())
        .collect::<BTreeSet<_>>();
    let rules = rules
        .iter()
        .map(|(after, variants)| {
            (
                (*after).to_owned(),
                variants
                    .iter()
                    .map(|(left, right)| RuleVariant::new(*left, *right))
                    .collect(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    Arc::new(Resources::from_memory(
        forms,
        SandhiRules::from_memory(rules),
        None,
    ))
}

fn scored_resources(
    forms: &[&str],
    rules: &[(&str, &[(&str, &str)])],
    scorer: Arc<dyn SequenceScorer>,
) -> Arc<Resources> {
    let forms = forms
        .iter()
        .map(|word| (*word).to_owned())
        .collect::<BTreeSet<_>>();
    Arc::new(Resources::from_memory(
        forms,
        SandhiRules::from_memory(
            rules
                .iter()
                .map(|(after, variants)| {
                    (
                        (*after).to_owned(),
                        variants
                            .iter()
                            .map(|(left, right)| RuleVariant::new(*left, *right))
                            .collect(),
                    )
                })
                .collect(),
        ),
        Some(scorer),
    ))
}

#[test]
fn rules_expand_every_position_and_honor_beginning_markers() {
    let rules = SandhiRules::from_memory(BTreeMap::from([
        (
            "ab".to_owned(),
            vec![RuleVariant::new("x", "y"), RuleVariant::new("^z", "q")],
        ),
        ("b".to_owned(), vec![RuleVariant::new("m", "n")]),
    ]));

    let splits = rules.split_all("ab", 0, None).unwrap();

    assert!(splits.contains(&("x".to_owned(), "y".to_owned())));
    assert!(splits.contains(&("z".to_owned(), "q".to_owned())));
    assert!(splits.contains(&("am".to_owned(), "n".to_owned())));
    assert!(!rules
        .split_all("cab", 0, None)
        .unwrap()
        .contains(&("cz".to_owned(), "q".to_owned())));
}

#[test]
fn splitter_builds_all_valid_recursive_paths() {
    let resources = resources(
        &["a", "b", "c", "ab"],
        &[
            ("abc", &[("a", "bc"), ("ab", "c")]),
            ("bc", &[("b", "c")]),
            ("c", &[("c", "")]),
        ],
    );
    let splitter = Splitter::new(resources);

    let paths = splitter
        .split(
            "abc",
            SplitOptions {
                limit: 10,
                scored: false,
            },
        )
        .unwrap()
        .unwrap();

    assert_eq!(paths, vec![vec!["ab", "c"], vec!["a", "b", "c"]]);
}

#[test]
fn adversarial_split_depth_fails_loudly_instead_of_overflowing_the_stack() {
    let forms = BTreeSet::from(["a".to_owned()]);
    let rules = (0..=512)
        .map(|index| {
            let state = format!("S{index:03}");
            let right = if index == 512 {
                String::new()
            } else {
                format!("S{:03}", index + 1)
            };
            (state, vec![RuleVariant::new("a", right)])
        })
        .collect();
    let splitter = Splitter::new(Arc::new(Resources::from_memory(
        forms,
        SandhiRules::from_memory(rules),
        None,
    )));

    assert!(matches!(
        splitter.split(
            "S000",
            SplitOptions {
                limit: 10,
                scored: false,
            }
        ),
        Err(CoreError::GraphDepthLimit { max_depth: 512 })
    ));
}

#[test]
fn deepest_packaged_benchmark_compound_remains_below_the_safety_limit() {
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../process_sanskrit/splitter/data/native");
    let resources = Arc::new(Resources::load_unscored(data_dir).unwrap());
    let text = concat!(
        "BUtapretavetAlaqAkinIjvaradagDakaRqUkiwIBakuzwapittakaplIhaBagaMdaralUtApAmA",
        "vEsarpalohaliNgASezaSvAsatrAsakAsamUrCAgaravizayayogAgnyudakamAramArIkalaha",
        "vErakAntArAkAlamftyutryambukatrElAwakavfScikasarpanakulasiMhavyAGrarkzatarakzu",
        "carmaramakaravfkataskarAjIvakAyikAnapanayantu",
    );

    let result = Splitter::new(resources).split(
        text,
        SplitOptions {
            limit: 10,
            scored: false,
        },
    );

    assert!(!matches!(result, Err(CoreError::GraphDepthLimit { .. })));
}

#[test]
fn no_valid_root_is_distinct_from_an_empty_result_limit() {
    let splitter = Splitter::new(resources(&["a"], &[("z", &[("z", "")])]));
    assert_eq!(
        splitter
            .split(
                "z",
                SplitOptions {
                    limit: 10,
                    scored: false,
                },
            )
            .unwrap(),
        None
    );

    let splitter = Splitter::new(resources(&["a"], &[("a", &[("a", "")])]));
    assert_eq!(
        splitter
            .split(
                "a",
                SplitOptions {
                    limit: 0,
                    scored: false,
                },
            )
            .unwrap(),
        Some(Vec::new())
    );
}

#[test]
fn scored_zero_limit_preserves_the_python_value_error_contract() {
    let splitter = Splitter::new(scored_resources(
        &["a"],
        &[("a", &[("a", "")])],
        Arc::new(ConstantScorer),
    ));
    assert!(matches!(
        splitter.split(
            "a",
            SplitOptions {
                limit: 0,
                scored: true,
            }
        ),
        Err(CoreError::InvalidLimit { limit: 0 })
    ));
}

#[test]
fn scored_zero_limit_surfaces_scorer_failure_first() {
    let splitter = Splitter::new(scored_resources(
        &["a"],
        &[("a", &[("a", "")])],
        Arc::new(RejectingScorer),
    ));
    assert!(matches!(
        splitter.split(
            "a",
            SplitOptions {
                limit: 0,
                scored: true,
            }
        ),
        Err(CoreError::ScorerUnavailable(message))
            if message.contains("must fail before limit validation")
    ));
}

#[test]
fn hostile_variant_group_count_is_rejected_before_allocation() {
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../process_sanskrit/splitter/data/native");
    let temp = tempfile::tempdir().unwrap();
    let afters = temp.path().join("sandhi_after.fst");
    let variants = temp.path().join("sandhi_variants.bin");
    fs::copy(data_dir.join("sandhi_after.fst"), &afters).unwrap();

    let mut hostile = Vec::from(&b"PSSV0001"[..]);
    hostile.extend_from_slice(&1_u32.to_le_bytes());
    hostile.extend_from_slice(&u32::MAX.to_le_bytes());
    hostile.extend_from_slice(&0_u64.to_le_bytes());
    fs::write(&variants, hostile).unwrap();

    let error = SandhiRules::load(&afters, &variants).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("group offset table exceeds file length"),
        "unexpected error: {error}"
    );
}

#[test]
fn missing_scorer_never_degrades_to_length_ranking() {
    let splitter = Splitter::new(resources(&["a"], &[("a", &[("a", "")])]));
    for limit in [10, 1001] {
        assert!(matches!(
            splitter.split(
                "a",
                SplitOptions {
                    limit,
                    scored: true,
                }
            ),
            Err(CoreError::ScorerUnavailable(_))
        ));
    }

    let _encoder: Arc<dyn PieceEncoder> = Arc::new(RejectingEncoder);
}

#[derive(Debug)]
struct TwoStageScorer;

impl SequenceScorer for TwoStageScorer {
    fn score_sequence(&self, tokens: &[&str]) -> Result<f32, CoreError> {
        Ok(match tokens.join(" ").as_str() {
            // Edge scores put the a/b/c route into the shortlist first.
            "a b" | "b c" => -1.0,
            "x y" | "y z" => -2.0,
            // Complete-sequence scoring then makes x/y/z the winner.
            "a b c" => -20.0,
            "x y z" => -5.0,
            _ => -1.0,
        })
    }
}

#[derive(Debug)]
struct ConstantScorer;

impl SequenceScorer for ConstantScorer {
    fn score_sequence(&self, _tokens: &[&str]) -> Result<f32, CoreError> {
        Ok(-1.0)
    }
}

#[test]
fn full_sequence_scoring_stably_reranks_the_edge_shortlist() {
    let resources = scored_resources(
        &["a", "b", "c", "x", "y", "z"],
        &[
            ("R", &[("a", "B"), ("x", "Y")]),
            ("B", &[("b", "C")]),
            ("C", &[("c", "")]),
            ("Y", &[("y", "Z")]),
            ("Z", &[("z", "")]),
        ],
        Arc::new(TwoStageScorer),
    );

    let paths = Splitter::new(resources)
        .split(
            "R",
            SplitOptions {
                limit: 2,
                scored: true,
            },
        )
        .unwrap()
        .unwrap();

    assert_eq!(paths[0], ["x", "y", "z"]);
    assert_eq!(paths[1], ["a", "b", "c"]);
}

#[test]
fn packaged_native_assets_load_and_split_a_known_compound() {
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../process_sanskrit/splitter/data/native");
    let resources = Arc::new(Resources::load_unscored(data_dir).unwrap());
    assert!(resources.valid("asti"));
    assert!(!resources.valid("xyzzy"));

    let paths = Splitter::new(resources)
        .split(
            "astyuttarasyAMdiSi",
            SplitOptions {
                limit: 10,
                scored: false,
            },
        )
        .unwrap()
        .unwrap();

    assert!(paths.contains(&vec![
        "asti".to_owned(),
        "uttarasyAm".to_owned(),
        "diSi".to_owned(),
    ]));
}

#[test]
fn loaded_native_graph_assets_survive_later_package_file_mutation() {
    let source_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../process_sanskrit/splitter/data/native");
    let temp = tempfile::tempdir().unwrap();
    for name in [
        "native-assets.json",
        "forms.fst",
        "sandhi_after.fst",
        "sandhi_variants.bin",
    ] {
        fs::copy(source_dir.join(name), temp.path().join(name)).unwrap();
    }

    let resources = Resources::load_unscored(temp.path()).unwrap();
    for name in ["forms.fst", "sandhi_after.fst", "sandhi_variants.bin"] {
        let path = temp.path().join(name);
        let len = fs::metadata(&path).unwrap().len() as usize;
        fs::write(path, vec![0_u8; len]).unwrap();
    }

    let still_valid = resources.valid("asti");
    let splits = resources
        .rules()
        .split_all("astyuttarasyAMdiSi", 0, None)
        .unwrap();

    assert!(still_valid);
    assert!(splits.contains(&("asti".to_owned(), "uttarasyAMdiSi".to_owned())));
}

#[test]
fn encoder_factory_receives_the_verified_sentencepiece_bytes() {
    let source_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../process_sanskrit/splitter/data/native");
    let temp = tempfile::tempdir().unwrap();
    for name in [
        "native-assets.json",
        "forms.fst",
        "sandhi_after.fst",
        "sandhi_variants.bin",
        "scorer.bin",
        "scorer_vocab.fst",
        "sentencepiece.model",
    ] {
        fs::copy(source_dir.join(name), temp.path().join(name)).unwrap();
    }
    let expected = fs::read(source_dir.join("sentencepiece.model")).unwrap();
    let model_path = temp.path().join("sentencepiece.model");

    let resources = Resources::load_with_encoder_factory(temp.path(), |verified| {
        fs::write(&model_path, b"mutated after verification").unwrap();
        assert_eq!(verified, expected.as_slice());
        Ok(Arc::new(RejectingEncoder) as Arc<dyn PieceEncoder>)
    })
    .unwrap();

    assert!(resources.valid("asti"));
}

#[test]
fn shared_splitter_is_safe_for_concurrent_requests() {
    let resources = resources(
        &["a", "b", "c", "ab"],
        &[
            ("abc", &[("a", "bc"), ("ab", "c")]),
            ("bc", &[("b", "c")]),
            ("c", &[("c", "")]),
        ],
    );
    let splitter = Arc::new(Splitter::new(resources));
    let expected = splitter
        .split(
            "abc",
            SplitOptions {
                limit: 10,
                scored: false,
            },
        )
        .unwrap();

    let threads = (0..8)
        .map(|_| {
            let splitter = Arc::clone(&splitter);
            let expected = expected.clone();
            std::thread::spawn(move || {
                for _ in 0..100 {
                    assert_eq!(
                        splitter
                            .split(
                                "abc",
                                SplitOptions {
                                    limit: 10,
                                    scored: false,
                                },
                            )
                            .unwrap(),
                        expected
                    );
                }
            })
        })
        .collect::<Vec<_>>();
    for thread in threads {
        thread.join().unwrap();
    }
}

#[test]
fn tied_scores_use_canonical_token_order() {
    let resources = scored_resources(
        &["a", "b", "x", "y"],
        &[
            ("R", &[("x", "Y"), ("a", "B")]),
            ("B", &[("b", "")]),
            ("Y", &[("y", "")]),
        ],
        Arc::new(ConstantScorer),
    );
    let paths = Splitter::new(resources)
        .split(
            "R",
            SplitOptions {
                limit: 10,
                scored: true,
            },
        )
        .unwrap()
        .unwrap();

    assert_eq!(paths, vec![vec!["a", "b"], vec!["x", "y"]]);
}

#[test]
fn state_specific_nodes_do_not_duplicate_identical_output_sequences() {
    let resources = resources(
        &["a", "b"],
        &[
            ("R", &[("a", "B1"), ("a", "B2")]),
            ("B1", &[("b", "")]),
            ("B2", &[("b", "")]),
        ],
    );
    let paths = Splitter::new(resources)
        .split(
            "R",
            SplitOptions {
                limit: 10,
                scored: false,
            },
        )
        .unwrap()
        .unwrap();

    assert_eq!(paths, vec![vec!["a", "b"]]);
}
