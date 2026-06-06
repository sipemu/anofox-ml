use ndarray::{Array1, Array2};
use rustml_core::Float;
use std::path::Path;
use std::str::FromStr;

/// Result of reading a CSV file: (features, optional target, optional headers).
pub type CsvReadResult<F> = Result<(Array2<F>, Option<Array1<F>>, Option<Vec<String>>), CsvError>;

/// Options for reading CSV files.
#[derive(Debug, Clone)]
pub struct CsvReadOptions {
    /// Whether the first row is a header (default: true).
    pub has_header: bool,
    /// Field delimiter (default: b',').
    pub delimiter: u8,
    /// Column index to use as the target variable (y). If `None`, no target
    /// is extracted and only the feature matrix X is returned.
    pub target_column: Option<usize>,
}

impl Default for CsvReadOptions {
    fn default() -> Self {
        Self {
            has_header: true,
            delimiter: b',',
            target_column: None,
        }
    }
}

impl CsvReadOptions {
    /// Create default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether the file has a header row.
    pub fn with_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    /// Set the field delimiter.
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Set the target column index (for supervised learning).
    pub fn with_target_column(mut self, col: usize) -> Self {
        self.target_column = Some(col);
        self
    }
}

/// Parse a single CSV record into a vector of floats.
fn parse_record_to_floats<F: Float + FromStr>(
    record: &csv::StringRecord,
    row_idx: usize,
) -> Result<Vec<F>, CsvError> {
    record
        .iter()
        .enumerate()
        .map(|(col_idx, field)| {
            let trimmed = field.trim();
            F::from_str(trimmed).map_err(|_| {
                CsvError::Parse(format!(
                    "cannot parse '{}' as float at row {}, col {}",
                    trimmed, row_idx, col_idx
                ))
            })
        })
        .collect()
}

/// Validate that every row in `all_values` has exactly `n_cols` columns.
fn validate_column_consistency<F: Float>(
    all_values: &[Vec<F>],
    n_cols: usize,
) -> Result<(), CsvError> {
    for (i, row) in all_values.iter().enumerate() {
        if row.len() != n_cols {
            return Err(CsvError::Parse(format!(
                "row {} has {} columns, expected {}",
                i,
                row.len(),
                n_cols
            )));
        }
    }
    Ok(())
}

/// Split the parsed values into a feature matrix and target vector by
/// extracting the column at `target_col`.
fn split_features_and_target<F: Float>(
    all_values: Vec<Vec<F>>,
    target_col: usize,
    n_rows: usize,
    n_cols: usize,
) -> Result<(Array2<F>, Array1<F>), CsvError> {
    if target_col >= n_cols {
        return Err(CsvError::Parse(format!(
            "target_column {} out of range (file has {} columns)",
            target_col, n_cols
        )));
    }

    let feature_cols = n_cols - 1;
    let mut x_data = Vec::with_capacity(n_rows * feature_cols);
    let mut y_data = Vec::with_capacity(n_rows);

    for row in &all_values {
        y_data.push(row[target_col]);
        for (j, &val) in row.iter().enumerate() {
            if j != target_col {
                x_data.push(val);
            }
        }
    }

    let x = Array2::from_shape_vec((n_rows, feature_cols), x_data)
        .map_err(|e| CsvError::Parse(e.to_string()))?;
    let y = Array1::from_vec(y_data);

    Ok((x, y))
}

/// Parse header names from the reader, if configured.
fn parse_headers(
    reader: &mut csv::Reader<std::fs::File>,
    has_header: bool,
) -> Result<Option<Vec<String>>, CsvError> {
    if has_header {
        Ok(Some(
            reader
                .headers()
                .map_err(|e| CsvError::Io(e.to_string()))?
                .iter()
                .map(|s| s.to_string())
                .collect(),
        ))
    } else {
        Ok(None)
    }
}

/// Read all CSV records into a vector of parsed float rows.
fn read_all_records<F: Float + FromStr>(
    reader: &mut csv::Reader<std::fs::File>,
) -> Result<Vec<Vec<F>>, CsvError> {
    let mut all_values: Vec<Vec<F>> = Vec::new();
    for (row_idx, result) in reader.records().enumerate() {
        let record = result.map_err(|e| CsvError::Parse(format!("row {}: {}", row_idx, e)))?;
        all_values.push(parse_record_to_floats(&record, row_idx)?);
    }
    if all_values.is_empty() {
        return Err(CsvError::Empty);
    }
    Ok(all_values)
}

/// Validate columns and assemble the final result, optionally splitting
/// a target column out of the feature matrix.
fn assemble_result<F: Float>(
    all_values: Vec<Vec<F>>,
    target_column: Option<usize>,
    headers: Option<Vec<String>>,
) -> CsvReadResult<F> {
    let n_rows = all_values.len();
    let n_cols = all_values[0].len();
    validate_column_consistency(&all_values, n_cols)?;

    match target_column {
        Some(target_col) => {
            let (x, y) = split_features_and_target(all_values, target_col, n_rows, n_cols)?;
            Ok((x, Some(y), headers))
        }
        None => {
            let flat: Vec<F> = all_values.into_iter().flatten().collect();
            let x = Array2::from_shape_vec((n_rows, n_cols), flat)
                .map_err(|e| CsvError::Parse(e.to_string()))?;
            Ok((x, None, headers))
        }
    }
}

/// Read a CSV file into an ndarray feature matrix (and optionally a target vector).
///
/// Returns `(X, Option<y>, Option<header_names>)`.
///
/// - If `options.target_column` is set, that column is extracted as `y` and
///   excluded from `X`.
/// - If `options.has_header` is true, header names are returned.
pub fn read_csv<F, P>(path: P, options: &CsvReadOptions) -> CsvReadResult<F>
where
    F: Float + FromStr,
    P: AsRef<Path>,
{
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(options.has_header)
        .delimiter(options.delimiter)
        .from_path(path.as_ref())
        .map_err(|e| CsvError::Io(e.to_string()))?;

    let headers = parse_headers(&mut reader, options.has_header)?;
    let all_values = read_all_records(&mut reader)?;
    assemble_result(all_values, options.target_column, headers)
}

/// Convenience function: read a CSV file with headers, returning only the
/// feature matrix and target vector.
pub fn read_csv_with_header<F, P>(
    path: P,
    target_column: usize,
) -> Result<(Array2<F>, Array1<F>), CsvError>
where
    F: Float + FromStr,
    P: AsRef<Path>,
{
    let options = CsvReadOptions::new().with_target_column(target_column);
    let (x, y, _) = read_csv(path, &options)?;
    match y {
        Some(y) => Ok((x, y)),
        None => Err(CsvError::Parse("target_column should have been set".into())),
    }
}

/// Errors that can occur when reading CSV files.
#[derive(Debug)]
pub enum CsvError {
    /// I/O error (file not found, permission denied, etc.)
    Io(String),
    /// Parse error (invalid float, inconsistent columns, etc.)
    Parse(String),
    /// The CSV file is empty.
    Empty,
}

impl std::fmt::Display for CsvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CsvError::Io(msg) => write!(f, "CSV I/O error: {}", msg),
            CsvError::Parse(msg) => write!(f, "CSV parse error: {}", msg),
            CsvError::Empty => write!(f, "CSV file is empty"),
        }
    }
}

impl std::error::Error for CsvError {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::io::Write;

    fn write_temp_csv(content: &str) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_read_csv_basic() {
        let csv = "a,b,c\n1.0,2.0,3.0\n4.0,5.0,6.0\n";
        let file = write_temp_csv(csv);
        let options = CsvReadOptions::new();
        let (x, y, headers): (Array2<f64>, _, _) = read_csv(file.path(), &options).unwrap();

        assert_eq!(x.shape(), &[2, 3]);
        assert_abs_diff_eq!(x[[0, 0]], 1.0);
        assert_abs_diff_eq!(x[[1, 2]], 6.0);
        assert!(y.is_none());
        assert_eq!(headers.unwrap(), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_read_csv_with_target() {
        let csv = "f1,f2,label\n1.0,2.0,0.0\n3.0,4.0,1.0\n5.0,6.0,0.0\n";
        let file = write_temp_csv(csv);
        let options = CsvReadOptions::new().with_target_column(2);
        let (x, y, _): (Array2<f64>, _, _) = read_csv(file.path(), &options).unwrap();

        assert_eq!(x.shape(), &[3, 2]);
        assert_abs_diff_eq!(x[[0, 0]], 1.0);
        assert_abs_diff_eq!(x[[2, 1]], 6.0);

        let y = y.unwrap();
        assert_abs_diff_eq!(y[0], 0.0);
        assert_abs_diff_eq!(y[1], 1.0);
        assert_abs_diff_eq!(y[2], 0.0);
    }

    #[test]
    fn test_read_csv_no_header() {
        let csv = "1.0,2.0\n3.0,4.0\n";
        let file = write_temp_csv(csv);
        let options = CsvReadOptions::new().with_header(false);
        let (x, _, headers): (Array2<f64>, _, _) = read_csv(file.path(), &options).unwrap();

        assert_eq!(x.shape(), &[2, 2]);
        assert!(headers.is_none());
    }

    #[test]
    fn test_read_csv_semicolon_delimiter() {
        let csv = "a;b\n1.0;2.0\n3.0;4.0\n";
        let file = write_temp_csv(csv);
        let options = CsvReadOptions::new().with_delimiter(b';');
        let (x, _, _): (Array2<f64>, _, _) = read_csv(file.path(), &options).unwrap();

        assert_eq!(x.shape(), &[2, 2]);
        assert_abs_diff_eq!(x[[0, 1]], 2.0);
    }

    #[test]
    fn test_read_csv_convenience() {
        let csv = "f1,f2,y\n1.0,2.0,10.0\n3.0,4.0,20.0\n";
        let file = write_temp_csv(csv);
        let (x, y): (Array2<f64>, Array1<f64>) = read_csv_with_header(file.path(), 2).unwrap();

        assert_eq!(x.shape(), &[2, 2]);
        assert_abs_diff_eq!(y[0], 10.0);
        assert_abs_diff_eq!(y[1], 20.0);
    }

    #[test]
    fn test_read_csv_empty_file() {
        let csv = "a,b\n";
        let file = write_temp_csv(csv);
        let options = CsvReadOptions::new();
        let result: Result<(Array2<f64>, _, _), _> = read_csv(file.path(), &options);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_csv_parse_error() {
        let csv = "a,b\n1.0,not_a_number\n";
        let file = write_temp_csv(csv);
        let options = CsvReadOptions::new();
        let result: Result<(Array2<f64>, _, _), _> = read_csv(file.path(), &options);
        assert!(result.is_err());
    }
}
