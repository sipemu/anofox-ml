//! CSV data loading with ndarray integration.
//!
//! This crate provides functions for reading CSV files into
//! [`ndarray::Array2`] feature matrices and optional [`ndarray::Array1`]
//! target vectors. It supports configurable delimiters, optional headers,
//! and target column extraction.
//!
//! # Examples
//!
//! ```no_run
//! use rustml_io::{read_csv, CsvReadOptions};
//!
//! let options = CsvReadOptions::new()
//!     .with_target_column(2);
//!
//! let (x, y, headers) = read_csv::<f64, _>("data.csv", &options).unwrap();
//! // x: Array2<f64> -- feature matrix (all columns except column 2)
//! // y: Some(Array1<f64>) -- target vector (column 2)
//! // headers: Some(Vec<String>) -- column names from the first row
//! ```

pub mod csv_reader;

pub use csv_reader::{read_csv, read_csv_with_header, CsvError, CsvReadOptions, CsvReadResult};
