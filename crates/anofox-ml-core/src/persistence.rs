//! Model persistence: save and load fitted models in JSON or bincode format.
//!
//! All fitted model types in anofox-ml derive `Serialize` and `Deserialize`, so
//! they can be saved to disk and loaded back without re-training.
//!
//! # Examples
//!
//! ```no_run
//! use anofox_ml_core::persistence::{save_json, load_json};
//! # use serde::{Serialize, Deserialize};
//! # #[derive(Serialize, Deserialize)] struct MyModel { x: f64 }
//! # let model = MyModel { x: 1.0 };
//!
//! // Save to file
//! save_json(&model, "model.json").unwrap();
//!
//! // Load from file
//! let loaded: MyModel = load_json("model.json").unwrap();
//! ```

use std::path::Path;

use serde::{de::DeserializeOwned, Serialize};

use crate::{Result, RustMlError};

// ---------------------------------------------------------------------------
// JSON – file-based
// ---------------------------------------------------------------------------

/// Serialize a model to a JSON file.
pub fn save_json<T: Serialize>(model: &T, path: impl AsRef<Path>) -> Result<()> {
    let json = serde_json::to_string_pretty(model)
        .map_err(|e| RustMlError::Serialization(e.to_string()))?;
    std::fs::write(path, json).map_err(|e| RustMlError::Io(e.to_string()))
}

/// Deserialize a model from a JSON file.
pub fn load_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let data = std::fs::read_to_string(path).map_err(|e| RustMlError::Io(e.to_string()))?;
    serde_json::from_str(&data).map_err(|e| RustMlError::Serialization(e.to_string()))
}

// ---------------------------------------------------------------------------
// Bincode – file-based
// ---------------------------------------------------------------------------

/// Serialize a model to a bincode file (compact binary format).
pub fn save_bincode<T: Serialize>(model: &T, path: impl AsRef<Path>) -> Result<()> {
    let bytes = bincode::serialize(model).map_err(|e| RustMlError::Serialization(e.to_string()))?;
    std::fs::write(path, bytes).map_err(|e| RustMlError::Io(e.to_string()))
}

/// Deserialize a model from a bincode file.
pub fn load_bincode<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let data = std::fs::read(path).map_err(|e| RustMlError::Io(e.to_string()))?;
    bincode::deserialize(&data).map_err(|e| RustMlError::Serialization(e.to_string()))
}

// ---------------------------------------------------------------------------
// JSON – in-memory
// ---------------------------------------------------------------------------

/// Serialize a model to a JSON string.
pub fn to_json_string<T: Serialize>(model: &T) -> Result<String> {
    serde_json::to_string(model).map_err(|e| RustMlError::Serialization(e.to_string()))
}

/// Deserialize a model from a JSON string.
pub fn from_json_string<T: DeserializeOwned>(s: &str) -> Result<T> {
    serde_json::from_str(s).map_err(|e| RustMlError::Serialization(e.to_string()))
}

// ---------------------------------------------------------------------------
// Bincode – in-memory
// ---------------------------------------------------------------------------

/// Serialize a model to bincode bytes.
pub fn to_bincode_bytes<T: Serialize>(model: &T) -> Result<Vec<u8>> {
    bincode::serialize(model).map_err(|e| RustMlError::Serialization(e.to_string()))
}

/// Deserialize a model from bincode bytes.
pub fn from_bincode_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    bincode::deserialize(bytes).map_err(|e| RustMlError::Serialization(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::fs;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct DummyModel {
        weights: Vec<f64>,
        bias: f64,
        label: String,
    }

    fn sample_model() -> DummyModel {
        DummyModel {
            weights: vec![1.0, -2.5, 3.125],
            bias: 0.42,
            label: "test_model".to_string(),
        }
    }

    #[test]
    fn json_roundtrip_in_memory() {
        let model = sample_model();
        let json = to_json_string(&model).unwrap();
        let loaded: DummyModel = from_json_string(&json).unwrap();
        assert_eq!(model, loaded);
    }

    #[test]
    fn bincode_roundtrip_in_memory() {
        let model = sample_model();
        let bytes = to_bincode_bytes(&model).unwrap();
        let loaded: DummyModel = from_bincode_bytes(&bytes).unwrap();
        assert_eq!(model, loaded);
    }

    #[test]
    fn json_roundtrip_file() {
        let model = sample_model();
        let dir = tempfile("json_roundtrip_file");
        let path = dir.join("model.json");

        save_json(&model, &path).unwrap();
        let loaded: DummyModel = load_json(&path).unwrap();
        assert_eq!(model, loaded);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bincode_roundtrip_file() {
        let model = sample_model();
        let dir = tempfile("bincode_roundtrip_file");
        let path = dir.join("model.bin");

        save_bincode(&model, &path).unwrap();
        let loaded: DummyModel = load_bincode(&path).unwrap();
        assert_eq!(model, loaded);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_json_missing_file() {
        let result: std::result::Result<DummyModel, _> = load_json("/nonexistent/path.json");
        assert!(result.is_err());
    }

    #[test]
    fn load_bincode_missing_file() {
        let result: std::result::Result<DummyModel, _> = load_bincode("/nonexistent/path.bin");
        assert!(result.is_err());
    }

    /// Per-test temp directory. Must be unique across tests in this file
    /// because cargo runs tests in parallel threads of a single process, so
    /// keying only on PID causes races between file roundtrip tests.
    fn tempfile(test_name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("rustml_test_{}_{}", std::process::id(), test_name));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}
