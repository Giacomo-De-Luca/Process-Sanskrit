use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;
use std::ptr::{self, NonNull};
use std::sync::Arc;

use process_sanskrit_splitter_core::{CoreError, PieceEncoder, Resources, SplitOptions, Splitter};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[repr(C)]
struct PsSentencePiecePieces {
    data: *mut c_char,
    data_len: usize,
    offsets: *mut usize,
    len: usize,
    status: c_int,
    error: *mut c_char,
}

enum PsSentencePiece {}

unsafe extern "C" {
    fn ps_sentencepiece_create(
        model_data: *const c_char,
        model_len: usize,
        error: *mut *mut c_char,
    ) -> *mut PsSentencePiece;
    fn ps_sentencepiece_destroy(processor: *mut PsSentencePiece);
    fn ps_sentencepiece_encode(
        processor: *const PsSentencePiece,
        text: *const c_char,
        text_len: usize,
    ) -> PsSentencePiecePieces;
    fn ps_sentencepiece_pieces_destroy(result: PsSentencePiecePieces);
    fn ps_sentencepiece_error_destroy(error: *mut c_char);
}

struct SentencePieceEncoder {
    processor: NonNull<PsSentencePiece>,
}

impl std::fmt::Debug for SentencePieceEncoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SentencePieceEncoder(v0.2.1)")
    }
}

// The processor is immutable after model initialization. SentencePiece's
// Encode API is const and supports concurrent inference. Ownership remains in
// this Arc-held adapter until all split requests have finished.
unsafe impl Send for SentencePieceEncoder {}
unsafe impl Sync for SentencePieceEncoder {}

impl SentencePieceEncoder {
    fn load(model: &[u8]) -> Result<Self, CoreError> {
        let mut error = ptr::null_mut();
        // SAFETY: `model` remains valid for this call and `error` is a valid
        // out-pointer. C++ parses/copies the serialized model before returning.
        let processor = unsafe {
            ps_sentencepiece_create(model.as_ptr().cast::<c_char>(), model.len(), &mut error)
        };
        let Some(processor) = NonNull::new(processor) else {
            return Err(CoreError::ScorerUnavailable(take_error(
                error,
                "SentencePiece failed to load without an error message",
            )));
        };
        if !error.is_null() {
            // Defensive: a successful constructor must not leave an error.
            unsafe { ps_sentencepiece_error_destroy(error) };
        }
        Ok(Self { processor })
    }
}

impl Drop for SentencePieceEncoder {
    fn drop(&mut self) {
        // SAFETY: this adapter uniquely owns the processor allocation.
        unsafe { ps_sentencepiece_destroy(self.processor.as_ptr()) };
    }
}

impl PieceEncoder for SentencePieceEncoder {
    fn encode(&self, text: &str) -> Result<Vec<String>, CoreError> {
        // SAFETY: the input pointer is valid for `text.len()` bytes, and the
        // processor remains alive through this shared reference.
        let result = unsafe {
            ps_sentencepiece_encode(
                self.processor.as_ptr(),
                text.as_ptr().cast::<c_char>(),
                text.len(),
            )
        };
        let pieces = decode_pieces(&result);
        // SAFETY: decoding no longer borrows either returned allocation.
        unsafe { ps_sentencepiece_pieces_destroy(result) };
        pieces
    }
}

fn decode_pieces(result: &PsSentencePiecePieces) -> Result<Vec<String>, CoreError> {
    if result.status != 0 || !result.error.is_null() {
        let message = if result.error.is_null() {
            "SentencePiece encoding failed without an error message".to_owned()
        } else {
            // SAFETY: a non-null error is a NUL-terminated C allocation owned
            // by this result until ps_sentencepiece_pieces_destroy is called.
            unsafe { CStr::from_ptr(result.error) }
                .to_string_lossy()
                .into_owned()
        };
        return Err(CoreError::ScorerUnavailable(message));
    }
    if result.len == 0 {
        return Ok(Vec::new());
    }
    if result.offsets.is_null() || (result.data.is_null() && result.data_len != 0) {
        return Err(CoreError::ScorerUnavailable(
            "SentencePiece returned a malformed string buffer".into(),
        ));
    }
    // SAFETY: C++ allocated len + 1 offsets and data_len bytes.
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.len + 1) };
    let data = if result.data_len == 0 {
        &[][..]
    } else {
        // SAFETY: non-null data owns exactly data_len initialized bytes.
        unsafe { std::slice::from_raw_parts(result.data.cast::<u8>(), result.data_len) }
    };
    if offsets.first() != Some(&0)
        || offsets.last() != Some(&result.data_len)
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(CoreError::ScorerUnavailable(
            "SentencePiece returned invalid string offsets".into(),
        ));
    }
    offsets
        .windows(2)
        .map(|range| {
            std::str::from_utf8(&data[range[0]..range[1]])
                .map(str::to_owned)
                .map_err(|error| {
                    CoreError::ScorerUnavailable(format!(
                        "SentencePiece returned invalid UTF-8: {error}"
                    ))
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{decode_pieces, PsSentencePiecePieces};
    use std::ptr;

    #[test]
    fn ffi_failure_status_cannot_be_mistaken_for_empty_pieces() {
        let result = PsSentencePiecePieces {
            data: ptr::null_mut(),
            data_len: 0,
            offsets: ptr::null_mut(),
            len: 0,
            status: 1,
            error: ptr::null_mut(),
        };

        let error = decode_pieces(&result).unwrap_err();
        assert!(error
            .to_string()
            .contains("failed without an error message"));
    }
}

#[pyclass(module = "process_sanskrit.splitter._native")]
struct NativeSplitter {
    splitter: Splitter,
}

#[pymethods]
impl NativeSplitter {
    #[new]
    fn new(py: Python<'_>, data_dir: PathBuf) -> PyResult<Self> {
        let result = py.detach(move || {
            let resources = Arc::new(Resources::load_with_encoder_factory(&data_dir, |model| {
                Ok(Arc::new(SentencePieceEncoder::load(model)?) as Arc<dyn PieceEncoder>)
            })?);
            Ok::<_, CoreError>(Self {
                splitter: Splitter::new(resources),
            })
        });
        result.map_err(core_error_to_python)
    }

    #[pyo3(signature = (text, limit=10, scored=true))]
    fn split_slp1(
        &self,
        py: Python<'_>,
        text: String,
        limit: usize,
        scored: bool,
    ) -> PyResult<Option<Vec<Vec<String>>>> {
        py.detach(|| self.splitter.split(&text, SplitOptions { limit, scored }))
            .map_err(core_error_to_python)
    }

    fn valid_slp1(&self, word: &str) -> PyResult<bool> {
        self.splitter.valid_slp1(word).map_err(core_error_to_python)
    }

    fn score_slp1(&self, py: Python<'_>, sequences: Vec<Vec<String>>) -> PyResult<Vec<f32>> {
        py.detach(|| self.splitter.score_slp1(&sequences))
            .map_err(core_error_to_python)
    }
}

fn take_error(error: *mut c_char, fallback: &str) -> String {
    if error.is_null() {
        return fallback.to_owned();
    }
    // SAFETY: the C wrapper returns a NUL-terminated allocation.
    let message = unsafe { CStr::from_ptr(error) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: the error has not been freed elsewhere.
    unsafe { ps_sentencepiece_error_destroy(error) };
    message
}

fn core_error_to_python(error: CoreError) -> PyErr {
    match error {
        CoreError::InvalidSlp1(_) | CoreError::InvalidLimit { .. } => {
            PyValueError::new_err(error.to_string())
        }
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeSplitter>()?;
    module.add("ASSET_SCHEMA_VERSION", 1_u32)?;
    module.add("SENTENCEPIECE_VERSION", "0.2.1")?;
    module.add(
        "BUILD_PROFILE",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    Ok(())
}
