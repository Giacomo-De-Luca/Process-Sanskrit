//! Native sandhi splitting primitives.
//!
//! The crate deliberately accepts canonical SLP1 only. Transliteration,
//! normalization, warnings, and public `Split` construction remain in Python.

mod error;
mod resources;
mod rules;
mod scorer;
mod splitter;

pub use error::CoreError;
pub use resources::Resources;
pub use rules::{RuleVariant, SandhiRules};
pub use scorer::{DcsScorer, PieceEncoder, ScorerModel, SequenceScorer};
pub use splitter::{SplitOptions, Splitter};
