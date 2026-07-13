use std::path::PathBuf;

use thiserror::Error;

/// Failures surfaced by the native splitter.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("failed to access native splitter asset {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("native splitter asset {asset} is corrupt or incompatible: {message}")]
    AssetFormat { asset: String, message: String },

    #[error("native splitter asset {asset} failed its SHA-256 check")]
    ChecksumMismatch { asset: String },

    #[error("the native splitter accepts canonical ASCII SLP1 only: {0:?}")]
    InvalidSlp1(String),

    #[error("scored splitting with limit={limit} preserves Python's ValueError contract")]
    InvalidLimit { limit: usize },

    #[error("the sandhi split scorer is unavailable: {0}")]
    ScorerUnavailable(String),

    #[error("the sandhi graph contains a cycle at state {0:?}")]
    GraphCycle(String),

    #[error("the sandhi graph exceeds the safe traversal depth of {max_depth}")]
    GraphDepthLimit { max_depth: usize },
}

impl CoreError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn asset(asset: impl Into<String>, message: impl Into<String>) -> Self {
        Self::AssetFormat {
            asset: asset.into(),
            message: message.into(),
        }
    }
}
