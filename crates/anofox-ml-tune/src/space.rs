//! Search spaces: JSON parameter ranges mapped onto rustuna distributions.

use std::collections::BTreeMap;

use anofox_ml_core::{Result, RustMlError};
use rustuna_core::distribution::Distribution;
use rustuna_core::trial::Trial;
use serde_json::{Map, Value};

fn invalid(msg: impl Into<String>) -> RustMlError {
    RustMlError::InvalidParameter(msg.into())
}

/// One searched parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum Param {
    Int {
        low: i64,
        high: i64,
        step: i64,
        log: bool,
    },
    Float {
        low: f64,
        high: f64,
        step: Option<f64>,
        log: bool,
    },
    /// Any JSON values (numbers, strings, booleans, objects, null).
    Categorical(Vec<Value>),
}

/// Parameters to search, by estimator parameter name.
///
/// JSON form, one entry per parameter:
///
/// ```json
/// {
///   "max_depth":     {"int": [2, 12]},
///   "n_estimators":  {"int": [50, 500], "step": 50},
///   "learning_rate": {"float": [0.01, 0.3], "log": true},
///   "loss":          {"categorical": ["squared_error", "absolute_error"]}
/// }
/// ```
///
/// Bounds are inclusive. `log` samples on a log scale (bounds must be
/// positive); `step` discretises the range.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchSpace {
    params: BTreeMap<String, Param>,
}

impl SearchSpace {
    /// Parse the JSON form.
    pub fn from_json(space: &Value) -> Result<Self> {
        let obj = space
            .as_object()
            .ok_or_else(|| invalid("search space must be a JSON object"))?;
        if obj.is_empty() {
            return Err(invalid("search space is empty"));
        }
        let params = obj
            .iter()
            .map(|(name, spec)| Ok((name.clone(), parse_param(name, spec)?)))
            .collect::<Result<_>>()?;
        Ok(Self { params })
    }

    /// Parameter names, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.params.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.params.len()
    }

    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Ask `trial` for a value of every parameter, as estimator JSON values.
    pub(crate) fn suggest(&self, trial: &mut Trial) -> Result<Map<String, Value>> {
        let mut out = Map::new();
        for (name, param) in &self.params {
            let rustuna_err = |e: rustuna_core::Error| invalid(format!("suggesting '{name}': {e}"));
            let value = match param {
                Param::Int {
                    low,
                    high,
                    step,
                    log,
                } => {
                    let d = Distribution::new_int(*low, *high, *step, *log);
                    Value::from(trial.suggest(name, &d).map_err(rustuna_err)?.round() as i64)
                }
                Param::Float {
                    low,
                    high,
                    step,
                    log,
                } => {
                    let d = Distribution::new_float(*low, *high, *step, *log);
                    Value::from(trial.suggest(name, &d).map_err(rustuna_err)?)
                }
                Param::Categorical(choices) => {
                    let d = Distribution::new_categorical(choices.len());
                    let i = trial.suggest(name, &d).map_err(rustuna_err)? as usize;
                    choices[i].clone()
                }
            };
            out.insert(name.clone(), value);
        }
        Ok(out)
    }
}

fn parse_param(name: &str, spec: &Value) -> Result<Param> {
    let obj = spec.as_object().ok_or_else(|| {
        invalid(format!(
            "'{name}': expected {{\"int\": [lo, hi]}}, {{\"float\": [lo, hi]}} or {{\"categorical\": [...]}}"
        ))
    })?;
    let kinds: Vec<&str> = ["int", "float", "categorical"]
        .into_iter()
        .filter(|k| obj.contains_key(*k))
        .collect();
    let [kind] = kinds[..] else {
        return Err(invalid(format!(
            "'{name}' needs exactly one of \"int\", \"float\", \"categorical\""
        )));
    };
    let allowed: &[&str] = match kind {
        "categorical" => &["categorical"],
        _ => &[kind, "log", "step"],
    };
    if let Some(k) = obj.keys().find(|k| !allowed.contains(&k.as_str())) {
        return Err(invalid(format!("'{name}': unknown key '{k}'")));
    }
    let log = match obj.get("log") {
        None => false,
        Some(v) => v
            .as_bool()
            .ok_or_else(|| invalid(format!("'{name}': \"log\" must be a boolean")))?,
    };
    let bounds = |as_num: fn(&Value) -> Option<f64>| -> Result<(f64, f64)> {
        match obj[kind].as_array().map(Vec::as_slice) {
            Some([lo, hi]) => match (as_num(lo), as_num(hi)) {
                (Some(lo), Some(hi)) if lo <= hi => Ok((lo, hi)),
                (Some(_), Some(_)) => Err(invalid(format!("'{name}': low > high"))),
                _ => Err(invalid(format!("'{name}': bounds must be {kind} numbers"))),
            },
            _ => Err(invalid(format!("'{name}': \"{kind}\" must be [low, high]"))),
        }
    };
    let param = match kind {
        "int" => {
            let (low, high) = bounds(|v| v.as_i64().map(|i| i as f64))?;
            let step = match obj.get("step") {
                None => 1,
                Some(v) => v.as_i64().filter(|&s| s > 0).ok_or_else(|| {
                    invalid(format!("'{name}': \"step\" must be a positive integer"))
                })?,
            };
            if log && step != 1 {
                return Err(invalid(format!(
                    "'{name}': \"log\" and \"step\" can't be combined"
                )));
            }
            Param::Int {
                low: low as i64,
                high: high as i64,
                step,
                log,
            }
        }
        "float" => {
            let (low, high) = bounds(Value::as_f64)?;
            let step = match obj.get("step") {
                None | Some(Value::Null) => None,
                Some(v) => Some(
                    v.as_f64()
                        .filter(|&s| s > 0.0)
                        .ok_or_else(|| invalid(format!("'{name}': \"step\" must be positive")))?,
                ),
            };
            if log && step.is_some() {
                return Err(invalid(format!(
                    "'{name}': \"log\" and \"step\" can't be combined"
                )));
            }
            Param::Float {
                low,
                high,
                step,
                log,
            }
        }
        _ => {
            let choices = obj["categorical"]
                .as_array()
                .filter(|c| !c.is_empty())
                .ok_or_else(|| {
                    invalid(format!(
                        "'{name}': \"categorical\" must be a non-empty list"
                    ))
                })?;
            Param::Categorical(choices.clone())
        }
    };
    let positive = match &param {
        Param::Int { low, .. } => *low > 0,
        Param::Float { low, .. } => *low > 0.0,
        Param::Categorical(_) => true,
    };
    if log && !positive {
        return Err(invalid(format!(
            "'{name}': \"log\" needs a positive lower bound"
        )));
    }
    Ok(param)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_every_kind() {
        let s = SearchSpace::from_json(&json!({
            "a": {"int": [1, 10]},
            "b": {"float": [0.01, 1.0], "log": true},
            "c": {"categorical": ["x", 2, null]},
            "d": {"int": [0, 100], "step": 10},
        }))
        .unwrap();
        assert_eq!(s.names().collect::<Vec<_>>(), ["a", "b", "c", "d"]);
        assert_eq!(
            s.params["d"],
            Param::Int {
                low: 0,
                high: 100,
                step: 10,
                log: false
            }
        );
        assert_eq!(
            s.params["c"],
            Param::Categorical(vec![json!("x"), json!(2), Value::Null])
        );
    }

    #[test]
    fn rejects_malformed_specs() {
        for (spec, msg) in [
            (json!([]), "JSON object"),
            (json!({}), "empty"),
            (json!({"a": 3}), "expected"),
            (
                json!({"a": {"int": [1, 2], "float": [1, 2]}}),
                "exactly one",
            ),
            (json!({"a": {"int": [5, 2]}}), "low > high"),
            (json!({"a": {"int": [1.5, 2]}}), "int numbers"),
            (
                json!({"a": {"float": [0.0, 1.0], "log": true}}),
                "positive lower bound",
            ),
            (
                json!({"a": {"float": [0.1, 1.0], "log": true, "step": 0.1}}),
                "can't be combined",
            ),
            (json!({"a": {"categorical": []}}), "non-empty"),
            (
                json!({"a": {"int": [1, 2], "lg": true}}),
                "unknown key 'lg'",
            ),
        ] {
            let e = SearchSpace::from_json(&spec).unwrap_err().to_string();
            assert!(e.contains(msg), "{spec}: {e}");
        }
    }
}
