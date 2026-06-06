use thiserror::Error;

#[derive(Debug, Error)]
pub enum RustMlError {
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("Not fitted: {0}")]
    NotFitted(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Empty input: {0}")]
    EmptyInput(String),

    #[error("IO error: {0}")]
    Io(String),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

pub type Result<T> = std::result::Result<T, RustMlError>;
