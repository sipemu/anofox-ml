//! Model persistence: save and load fitted models in a compact binary format
//! or JSON.
//!
//! All fitted model types in anofox-ml derive `Serialize` and `Deserialize`, so
//! they can be saved to disk and loaded back without re-training.
//!
//! Every payload is wrapped in a versioned envelope that records the file
//! format version, the anofox-ml version that wrote it and the model's type
//! name. Loading checks the envelope first, so reading a model as the wrong
//! type or from a newer, incompatible format fails with a clear error
//! instead of producing garbage (bincode is not self-describing):
//!
//! - Binary (format version 2, [`to_binary_bytes`] / [`save_binary`]): the
//!   magic bytes `ANOFOXML`, the bincode-encoded header, then the payload as
//!   **MessagePack with field names, zstd-compressed** (with the zstd frame
//!   checksum). Field names make saved models survive changes to the fitted
//!   types: a field added with `#[serde(default)]` is filled in, a removed
//!   field is ignored, and fields or enum variants may be reordered (#87).
//! - JSON: `{"format": "anofox-ml", "format_version": 1, "crate_version":
//!   "…", "type_name": "…", "payload": …}`
//!
//! Reading also accepts binary format version 1 (the same header followed
//! by a bincode payload, written by development versions between 0.1 and
//! 0.2) and files written before the envelope existed (a bare bincode or
//! JSON payload, as anofox-ml 0.1 wrote them).
//! Version-1 payloads are positional, so they load only while the fitted
//! type's layout is unchanged.
//!
//! Type names are compared without module paths (`FittedPca<f64>`), so
//! moving a type between modules does not invalidate saved models.
//! Pipelines with boxed (`dyn`) steps cannot be persisted; persist their
//! fitted steps individually.
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

use std::io::Read;
use std::path::Path;

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{Result, RustMlError};

/// Binary format version written by [`to_binary_bytes`]: 2 = zstd-compressed
/// MessagePack with field names, 1 = bincode (read only).
pub const FORMAT_VERSION: u32 = 2;

/// Envelope version written into JSON files (the JSON layout is unchanged
/// since version 1).
pub const JSON_FORMAT_VERSION: u32 = 1;

/// Value of the envelope's `format` field.
pub const FORMAT_NAME: &str = "anofox-ml";

/// Leading bytes of an enveloped binary file.
pub const BINARY_MAGIC: &[u8; 8] = b"ANOFOXML";

/// Leading bytes of an enveloped binary file.
#[deprecated(since = "0.2.0", note = "renamed to BINARY_MAGIC")]
pub const BINCODE_MAGIC: &[u8; 8] = BINARY_MAGIC;

/// zstd compression level of format version 2.
const ZSTD_LEVEL: i32 = 3;

/// Envelope metadata stored in front of every payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelHeader {
    /// Always [`FORMAT_NAME`].
    pub format: String,
    /// Envelope layout version ([`FORMAT_VERSION`] when written).
    pub format_version: u32,
    /// Version of `anofox-ml-core` that wrote the file.
    pub crate_version: String,
    /// Type name of the payload, without module paths.
    pub type_name: String,
}

impl ModelHeader {
    /// Binary-format header for a payload of type `T`, written by this
    /// version.
    pub fn for_type<T: ?Sized>() -> Self {
        Self::with_version::<T>(FORMAT_VERSION)
    }

    fn with_version<T: ?Sized>(format_version: u32) -> Self {
        Self {
            format: FORMAT_NAME.to_string(),
            format_version,
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            type_name: short_type_name::<T>(),
        }
    }

    /// Check that a payload with this header can be read as `T`.
    pub fn validate<T: ?Sized>(&self) -> Result<()> {
        if self.format != FORMAT_NAME {
            return Err(RustMlError::Serialization(format!(
                "not an anofox-ml model file (format {:?})",
                self.format
            )));
        }
        if self.format_version > FORMAT_VERSION {
            return Err(RustMlError::Serialization(format!(
                "model file format {} (written by anofox-ml {}) is newer than this \
                 version supports ({})",
                self.format_version, self.crate_version, FORMAT_VERSION
            )));
        }
        let expected = short_type_name::<T>();
        if self.type_name != expected {
            return Err(RustMlError::Serialization(format!(
                "model file holds a {} (written by anofox-ml {}), not a {}",
                self.type_name, self.crate_version, expected
            )));
        }
        Ok(())
    }
}

/// `std::any::type_name` with every path reduced to its last segment, e.g.
/// `FittedPca<f64>` for `anofox_ml_preprocessing::pca::FittedPca<f64>`.
/// References are dropped (`&T` serializes exactly like `T`).
pub fn short_type_name<T: ?Sized>() -> String {
    let full = std::any::type_name::<T>()
        .replace("&mut ", "")
        .replace('&', "");
    let mut out = String::with_capacity(full.len());
    let mut token = String::new();
    let flush = |token: &mut String, out: &mut String| {
        out.push_str(token.rsplit("::").next().unwrap_or(""));
        token.clear();
    };
    for c in full.chars() {
        if c.is_alphanumeric() || c == '_' || c == ':' {
            token.push(c);
        } else {
            flush(&mut token, &mut out);
            out.push(c);
        }
    }
    flush(&mut token, &mut out);
    out
}

#[derive(Serialize)]
struct EnvelopeRef<'a, T> {
    #[serde(flatten)]
    header: ModelHeader,
    payload: &'a T,
}

fn ser_err(e: impl std::fmt::Display) -> RustMlError {
    RustMlError::Serialization(e.to_string())
}

// ---------------------------------------------------------------------------
// JSON – in-memory
// ---------------------------------------------------------------------------

/// Serialize a model to an enveloped JSON string.
pub fn to_json_string<T: Serialize>(model: &T) -> Result<String> {
    serde_json::to_string(&EnvelopeRef {
        header: ModelHeader::with_version::<T>(JSON_FORMAT_VERSION),
        payload: model,
    })
    .map_err(ser_err)
}

fn to_json_string_pretty<T: Serialize>(model: &T) -> Result<String> {
    serde_json::to_string_pretty(&EnvelopeRef {
        header: ModelHeader::with_version::<T>(JSON_FORMAT_VERSION),
        payload: model,
    })
    .map_err(ser_err)
}

/// Deserialize a model from a JSON string (enveloped, or a bare legacy
/// payload).
pub fn from_json_string<T: DeserializeOwned>(s: &str) -> Result<T> {
    let mut value: serde_json::Value = serde_json::from_str(s).map_err(ser_err)?;
    let is_envelope = value
        .get("format")
        .and_then(|f| f.as_str())
        .is_some_and(|f| f == FORMAT_NAME)
        && value.get("payload").is_some();
    if !is_envelope {
        return serde_json::from_value(value).map_err(ser_err);
    }
    let obj = value.as_object_mut().expect("envelope is an object");
    let payload = obj.remove("payload").expect("checked above");
    let header: ModelHeader =
        serde_json::from_value(serde_json::Value::Object(obj.clone())).map_err(ser_err)?;
    header.validate::<T>()?;
    serde_json::from_value(payload).map_err(ser_err)
}

/// Read only the envelope header of a JSON model string (`None` for a
/// legacy bare payload).
pub fn json_header(s: &str) -> Result<Option<ModelHeader>> {
    let mut value: serde_json::Value = serde_json::from_str(s).map_err(ser_err)?;
    match value.as_object_mut() {
        Some(obj) if obj.get("format").and_then(|f| f.as_str()) == Some(FORMAT_NAME) => {
            obj.remove("payload");
            Ok(Some(
                serde_json::from_value(serde_json::Value::Object(obj.clone())).map_err(ser_err)?,
            ))
        }
        _ => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Binary – in-memory
// ---------------------------------------------------------------------------

/// Serialize a model to enveloped binary bytes (format version 2:
/// zstd-compressed MessagePack with field names).
pub fn to_binary_bytes<T: Serialize>(model: &T) -> Result<Vec<u8>> {
    let mut out = BINARY_MAGIC.to_vec();
    bincode::serialize_into(&mut out, &ModelHeader::for_type::<T>()).map_err(ser_err)?;
    let mut encoder = zstd::stream::Encoder::new(&mut out, ZSTD_LEVEL).map_err(ser_err)?;
    encoder.include_checksum(true).map_err(ser_err)?;
    rmp_serde::encode::write_named(&mut encoder, model).map_err(ser_err)?;
    encoder.finish().map_err(ser_err)?;
    Ok(out)
}

/// Deserialize a model from binary bytes: format version 2 or 1, or a bare
/// legacy bincode payload.
pub fn from_binary_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    let Some(mut rest) = bytes.strip_prefix(BINARY_MAGIC.as_slice()) else {
        return bincode::deserialize(bytes).map_err(ser_err);
    };
    let header: ModelHeader = bincode::deserialize_from(&mut rest).map_err(ser_err)?;
    header.validate::<T>()?;
    let trailing = |n: usize| {
        RustMlError::Serialization(format!(
            "{n} trailing bytes after the {} payload",
            header.type_name
        ))
    };
    match header.format_version {
        1 => {
            let model = bincode::deserialize_from(&mut rest).map_err(ser_err)?;
            if !rest.is_empty() {
                return Err(trailing(rest.len()));
            }
            Ok(model)
        }
        2 => {
            let raw = zstd::stream::decode_all(rest).map_err(ser_err)?;
            let mut de = rmp_serde::Deserializer::new(std::io::Cursor::new(&raw[..]));
            let model = T::deserialize(&mut de).map_err(ser_err)?;
            let read = de.get_ref().position() as usize;
            if read != raw.len() {
                return Err(trailing(raw.len() - read));
            }
            Ok(model)
        }
        v => Err(RustMlError::Serialization(format!(
            "unknown binary model format version {v}"
        ))),
    }
}

/// Read only the envelope header of binary model bytes (`None` for a legacy
/// bare payload).
pub fn binary_header(bytes: &[u8]) -> Result<Option<ModelHeader>> {
    match bytes.strip_prefix(BINARY_MAGIC.as_slice()) {
        Some(mut rest) => Ok(Some(bincode::deserialize_from(&mut rest).map_err(ser_err)?)),
        None => Ok(None),
    }
}

/// Serialize a model to enveloped binary bytes.
#[deprecated(since = "0.2.0", note = "use to_binary_bytes (format version 2)")]
pub fn to_bincode_bytes<T: Serialize>(model: &T) -> Result<Vec<u8>> {
    to_binary_bytes(model)
}

/// Deserialize a model from binary bytes.
#[deprecated(since = "0.2.0", note = "use from_binary_bytes")]
pub fn from_bincode_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    from_binary_bytes(bytes)
}

/// Read only the envelope header of binary model bytes.
#[deprecated(since = "0.2.0", note = "use binary_header")]
pub fn bincode_header(bytes: &[u8]) -> Result<Option<ModelHeader>> {
    binary_header(bytes)
}

// ---------------------------------------------------------------------------
// File-based
// ---------------------------------------------------------------------------

/// Serialize a model to an enveloped JSON file.
pub fn save_json<T: Serialize>(model: &T, path: impl AsRef<Path>) -> Result<()> {
    let json = to_json_string_pretty(model)?;
    std::fs::write(path, json).map_err(|e| RustMlError::Io(e.to_string()))
}

/// Deserialize a model from a JSON file.
pub fn load_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let data = std::fs::read_to_string(path).map_err(|e| RustMlError::Io(e.to_string()))?;
    from_json_string(&data)
}

/// Serialize a model to an enveloped binary file (format version 2).
pub fn save_binary<T: Serialize>(model: &T, path: impl AsRef<Path>) -> Result<()> {
    let bytes = to_binary_bytes(model)?;
    std::fs::write(path, bytes).map_err(|e| RustMlError::Io(e.to_string()))
}

/// Deserialize a model from a binary file (format version 2 or 1).
pub fn load_binary<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let mut data = Vec::new();
    std::fs::File::open(path)
        .and_then(|mut f| f.read_to_end(&mut data))
        .map_err(|e| RustMlError::Io(e.to_string()))?;
    from_binary_bytes(&data)
}

/// Serialize a model to an enveloped binary file.
#[deprecated(since = "0.2.0", note = "use save_binary (format version 2)")]
pub fn save_bincode<T: Serialize>(model: &T, path: impl AsRef<Path>) -> Result<()> {
    save_binary(model, path)
}

/// Deserialize a model from a binary file.
#[deprecated(since = "0.2.0", note = "use load_binary")]
pub fn load_bincode<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    load_binary(path)
}

/// An estimator's configuration persisted next to its fitted state, so a
/// reloaded model reports the hyperparameters it was trained with.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBundle<C, M> {
    /// The unfitted estimator (hyperparameters).
    pub config: C,
    /// The fitted model.
    pub fitted: M,
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
    fn binary_roundtrip_in_memory() {
        let model = sample_model();
        let bytes = to_binary_bytes(&model).unwrap();
        let loaded: DummyModel = from_binary_bytes(&bytes).unwrap();
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
    fn binary_roundtrip_file() {
        let model = sample_model();
        let dir = tempfile("binary_roundtrip_file");
        let path = dir.join("model.bin");

        save_binary(&model, &path).unwrap();
        let loaded: DummyModel = load_binary(&path).unwrap();
        assert_eq!(model, loaded);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn envelope_is_written_and_validated() {
        let model = sample_model();
        let json = to_json_string(&model).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["format"], "anofox-ml");
        assert_eq!(v["format_version"], JSON_FORMAT_VERSION);
        assert_eq!(v["type_name"], "DummyModel");
        assert_eq!(v["crate_version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(json_header(&json).unwrap().unwrap().type_name, "DummyModel");

        // Wrong target type: a clear error from the header, not garbage.
        let err = from_json_string::<Vec<f64>>(&json).unwrap_err().to_string();
        assert!(err.contains("holds a DummyModel"), "{err}");
        let bytes = to_binary_bytes(&model).unwrap();
        assert!(bytes.starts_with(BINARY_MAGIC));
        assert_eq!(
            binary_header(&bytes).unwrap().unwrap().type_name,
            "DummyModel"
        );
        let err = from_binary_bytes::<Vec<f64>>(&bytes)
            .unwrap_err()
            .to_string();
        assert!(err.contains("holds a DummyModel"), "{err}");

        // Newer format versions are rejected.
        let mut v2 = v.clone();
        v2["format_version"] = serde_json::json!(FORMAT_VERSION + 1);
        let err = from_json_string::<DummyModel>(&v2.to_string())
            .unwrap_err()
            .to_string();
        assert!(err.contains("newer"), "{err}");

        // Trailing bytes are rejected.
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(from_binary_bytes::<DummyModel>(&longer).is_err());
    }

    #[test]
    fn legacy_bare_payloads_still_load() {
        let model = sample_model();
        let json = serde_json::to_string(&model).unwrap();
        assert_eq!(from_json_string::<DummyModel>(&json).unwrap(), model);
        assert!(json_header(&json).unwrap().is_none());
        let bytes = bincode::serialize(&model).unwrap();
        assert_eq!(from_binary_bytes::<DummyModel>(&bytes).unwrap(), model);
        assert!(binary_header(&bytes).unwrap().is_none());
    }

    #[test]
    fn short_type_names_drop_module_paths() {
        assert_eq!(short_type_name::<DummyModel>(), "DummyModel");
        assert_eq!(
            short_type_name::<ModelBundle<Vec<f64>, std::collections::HashMap<String, u8>>>(),
            "ModelBundle<Vec<f64>, HashMap<String, u8>>"
        );
        assert_eq!(
            short_type_name::<ModelBundle<&Vec<f64>, &mut DummyModel>>(),
            short_type_name::<ModelBundle<Vec<f64>, DummyModel>>()
        );
    }

    #[test]
    fn load_json_missing_file() {
        let result: std::result::Result<DummyModel, _> = load_json("/nonexistent/path.json");
        assert!(result.is_err());
    }

    #[test]
    fn load_binary_missing_file() {
        let result: std::result::Result<DummyModel, _> = load_binary("/nonexistent/path.bin");
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

    /// Bytes as written before 0.2.0 (binary format version 1).
    fn v1_bytes<T: Serialize>(model: &T) -> Vec<u8> {
        let mut out = BINARY_MAGIC.to_vec();
        bincode::serialize_into(&mut out, &ModelHeader::with_version::<T>(1)).unwrap();
        bincode::serialize_into(&mut out, model).unwrap();
        out
    }

    #[test]
    fn binary_writes_format_version_2() {
        let bytes = to_binary_bytes(&sample_model()).unwrap();
        let header = binary_header(&bytes).unwrap().unwrap();
        assert_eq!(header.format_version, 2);
        assert_eq!(FORMAT_VERSION, 2);
    }

    #[test]
    fn binary_format_version_1_still_loads() {
        let model = sample_model();
        let bytes = v1_bytes(&model);
        assert_eq!(binary_header(&bytes).unwrap().unwrap().format_version, 1);
        assert_eq!(from_binary_bytes::<DummyModel>(&bytes).unwrap(), model);
        let mut longer = bytes;
        longer.push(0);
        assert!(from_binary_bytes::<DummyModel>(&longer).is_err());
    }

    #[test]
    fn corrupted_payload_is_detected() {
        let mut bytes = to_binary_bytes(&sample_model()).unwrap();
        let last = bytes.len() - 6;
        bytes[last] ^= 0xFF;
        assert!(from_binary_bytes::<DummyModel>(&bytes).is_err());
    }

    #[test]
    fn unknown_binary_version_is_rejected() {
        let mut out = BINARY_MAGIC.to_vec();
        bincode::serialize_into(&mut out, &ModelHeader::with_version::<DummyModel>(3)).unwrap();
        let err = from_binary_bytes::<DummyModel>(&out)
            .unwrap_err()
            .to_string();
        assert!(err.contains("newer"), "{err}");
    }

    // Schema evolution: the same type name, a changed definition.
    mod before {
        #[derive(serde::Serialize)]
        pub struct Model {
            pub a: f64,
            pub b: Vec<f64>,
            pub kind: super::before_kind::Kind,
            pub dropped: String,
        }
    }
    mod before_kind {
        #[derive(serde::Serialize)]
        #[allow(dead_code)]
        pub enum Kind {
            First,
            Second(f64),
        }
    }
    mod after {
        #[derive(Debug, PartialEq, serde::Deserialize)]
        pub struct Model {
            // Reordered fields, `dropped` removed, `added` with a default.
            pub kind: super::after_kind::Kind,
            pub b: Vec<f64>,
            pub a: f64,
            #[serde(default)]
            pub added: Option<u32>,
        }
    }
    mod after_kind {
        #[derive(Debug, PartialEq, serde::Deserialize)]
        pub enum Kind {
            // Reordered variants and a new one.
            Third,
            Second(f64),
            First,
        }
    }

    #[test]
    fn format_version_2_survives_schema_changes() {
        let old = before::Model {
            a: 1.5,
            b: vec![2.0, -3.0],
            kind: before_kind::Kind::Second(0.25),
            dropped: "gone".into(),
        };
        let bytes = to_binary_bytes(&old).unwrap();
        let new: after::Model = from_binary_bytes(&bytes).unwrap();
        assert_eq!(
            new,
            after::Model {
                kind: after_kind::Kind::Second(0.25),
                b: vec![2.0, -3.0],
                a: 1.5,
                added: None,
            }
        );
        // The same change breaks a version-1 (bincode) file.
        assert!(from_binary_bytes::<after::Model>(&v1_bytes(&old)).is_err());
    }
}
