//! CatBoost's text features (`text_features`): the `ByDelimiter`
//! tokenizer, the frequency dictionaries, bag of words and the online
//! NaiveBayes / BM25 calcers.

use anofox_ml_core::{Result, RustMlError};
use ndarray::{Array2, ArrayView2};
use std::collections::{HashMap, HashSet};

/// CatBoost's `text_processing` options, one processing for every text
/// column (catboost's `feature_processing["default"]`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CatBoostTextOptions {
    /// The tokenizer's delimiter (`ByDelimiter`; empty tokens are skipped).
    pub delimiter: String,
    /// Lowercase the tokens (the tokenizer's `lowercasing`).
    pub lowercasing: bool,
    pub dictionaries: Vec<CatBoostTextDictionary>,
    /// Which calcers run on which dictionaries.
    pub processing: Vec<CatBoostTextProcessing>,
}

/// A word frequency dictionary (`TDictionaryBuilder`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CatBoostTextDictionary {
    /// Its `dictionary_id`, as named in [`CatBoostTextProcessing`].
    pub name: String,
    /// Consecutive words per token (`gram_order`).
    pub gram_order: usize,
    /// Tokens seen fewer times on the learn rows are dropped
    /// (`occurrence_lower_bound`).
    pub occurrence_lower_bound: u64,
    /// Keep the most frequent tokens only (`max_dictionary_size`).
    pub max_dictionary_size: Option<usize>,
}

/// Calcers over dictionaries (one `feature_processing` entry).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CatBoostTextProcessing {
    pub dictionaries: Vec<String>,
    pub calcers: Vec<CatBoostTextCalcer>,
}

/// CatBoost's text feature calcers.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CatBoostTextCalcer {
    /// A binary feature per dictionary token, for the `top_tokens_count`
    /// most frequent ones (catboost's default 2000).
    BoW { top_tokens_count: usize },
    /// The class probabilities of a multinomial naive Bayes, online
    /// (classification only).
    NaiveBayes,
    /// BM25 scores per class, online (classification only).
    Bm25,
}

impl CatBoostTextOptions {
    /// CatBoost's default processing: `BoW` over a `BiGram` and a `Word`
    /// dictionary (occurrence bound 3, the Python package's), plus
    /// `NaiveBayes` over `Word` for classification.
    pub fn default_for(classification: bool) -> Self {
        let dictionary = |name: &str, gram_order| CatBoostTextDictionary {
            name: name.into(),
            gram_order,
            occurrence_lower_bound: 3,
            max_dictionary_size: None,
        };
        let mut processing = vec![CatBoostTextProcessing {
            dictionaries: vec!["BiGram".into(), "Word".into()],
            calcers: vec![CatBoostTextCalcer::BoW {
                top_tokens_count: 2000,
            }],
        }];
        if classification {
            processing.push(CatBoostTextProcessing {
                dictionaries: vec!["Word".into()],
                calcers: vec![CatBoostTextCalcer::NaiveBayes],
            });
        }
        Self {
            delimiter: " ".into(),
            lowercasing: false,
            dictionaries: vec![dictionary("BiGram", 2), dictionary("Word", 1)],
            processing,
        }
    }

    fn validate(&self, classification: bool) -> Result<()> {
        let invalid = |m: String| Err(RustMlError::InvalidParameter(m));
        if self.delimiter.is_empty() {
            return invalid("text options: the delimiter is empty".into());
        }
        let mut names = HashSet::new();
        for d in &self.dictionaries {
            if !names.insert(d.name.as_str()) {
                return invalid(format!("text options: dictionary '{}' twice", d.name));
            }
            if d.gram_order == 0 || d.max_dictionary_size == Some(0) {
                return invalid(format!(
                    "text options: dictionary '{}' needs gram_order and max_dictionary_size > 0",
                    d.name
                ));
            }
        }
        if self.processing.is_empty() {
            return invalid("text options: no feature processing".into());
        }
        for p in &self.processing {
            for name in &p.dictionaries {
                if !names.contains(name.as_str()) {
                    return invalid(format!("text options: unknown dictionary '{name}'"));
                }
            }
            for c in &p.calcers {
                match c {
                    CatBoostTextCalcer::BoW {
                        top_tokens_count: 0,
                    } => return invalid("text options: BoW needs top_tokens_count > 0".into()),
                    CatBoostTextCalcer::NaiveBayes | CatBoostTextCalcer::Bm25
                        if !classification =>
                    {
                        return invalid(format!(
                            "text options: {c:?} is supported only for classification"
                        ))
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
}

/// An online text calcer (CatBoost's online estimators).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) enum OnlineKind {
    NaiveBayes,
    Bm25,
}

/// One tokenized feature (a text column under one dictionary, CatBoost's
/// `TTokenizedFeatureDescription`) and its calcers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TokenizedFeature {
    column: usize,
    gram_order: usize,
    /// The dictionary's tokens in id order (words joined by `\0`).
    tokens: Vec<String>,
    /// Its BoW calcers: the number of features of each.
    bow: Vec<usize>,
    /// Its online calcers (NaiveBayes before BM25, as CatBoost).
    online: Vec<OnlineKind>,
}

/// Fitted text processing: the tokenizer and the tokenized features in
/// CatBoost's order (per text column, its dictionaries in first-use order).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct TextProcessing {
    delimiter: String,
    lowercasing: bool,
    n_columns: usize,
    features: Vec<TokenizedFeature>,
}

/// The token ids of a text with counts, sorted by id (CatBoost's `TText`).
pub(crate) type TokenCounts = Vec<(u32, u32)>;

impl TextProcessing {
    /// Build the dictionaries of every text column on the learn rows.
    pub(crate) fn fit(
        options: &CatBoostTextOptions,
        text: &[Vec<String>],
        classification: bool,
    ) -> Result<Self> {
        options.validate(classification)?;
        let mut out = Self {
            delimiter: options.delimiter.clone(),
            lowercasing: options.lowercasing,
            n_columns: text.len(),
            features: Vec::new(),
        };
        for (column, col) in text.iter().enumerate() {
            // The column's (dictionary, calcers) in first-use order
            // (`CreateTokenizedFeatures`).
            let mut uses: Vec<(&str, Vec<CatBoostTextCalcer>)> = Vec::new();
            for p in &options.processing {
                for name in &p.dictionaries {
                    let at = match uses.iter().position(|(n, _)| n == name) {
                        Some(at) => at,
                        None => {
                            uses.push((name, Vec::new()));
                            uses.len() - 1
                        }
                    };
                    uses[at].1.extend(p.calcers.iter().copied());
                }
            }
            let tokenized: Vec<Vec<String>> = col.iter().map(|t| out.tokenize(t)).collect();
            for (name, calcers) in uses {
                let d = options
                    .dictionaries
                    .iter()
                    .find(|d| d.name == name)
                    .expect("validated");
                let tokens = build_dictionary(&tokenized, d);
                let mut bow = Vec::new();
                let (mut nb, mut bm25) = (false, false);
                for c in calcers {
                    match c {
                        CatBoostTextCalcer::BoW { top_tokens_count } => {
                            if tokens.is_empty() {
                                return Err(RustMlError::InvalidParameter(format!(
                                    "text column {column}: dictionary '{name}' is empty, \
                                     decrease its occurrence_lower_bound"
                                )));
                            }
                            bow.push(top_tokens_count.min(tokens.len()));
                        }
                        CatBoostTextCalcer::NaiveBayes => nb = true,
                        CatBoostTextCalcer::Bm25 => bm25 = true,
                    }
                }
                let online = [(nb, OnlineKind::NaiveBayes), (bm25, OnlineKind::Bm25)]
                    .into_iter()
                    .filter_map(|(on, k)| on.then_some(k))
                    .collect();
                out.features.push(TokenizedFeature {
                    column,
                    gram_order: d.gram_order,
                    tokens,
                    bow,
                    online,
                });
            }
        }
        Ok(out)
    }

    /// The tokens of a text (`ByDelimiter`, empty ones skipped).
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split(self.delimiter.as_str())
            .filter(|t| !t.is_empty())
            .map(|t| {
                if self.lowercasing {
                    t.to_lowercase()
                } else {
                    t.to_string()
                }
            })
            .collect()
    }

    /// The number of BoW features.
    pub(crate) fn n_bow(&self) -> usize {
        self.features.iter().flat_map(|f| &f.bow).sum()
    }

    pub(crate) fn n_columns(&self) -> usize {
        self.n_columns
    }

    /// The online calcers in CatBoost's order: (tokenized feature, kind).
    pub(crate) fn online(&self) -> Vec<(usize, OnlineKind)> {
        self.features
            .iter()
            .enumerate()
            .flat_map(|(i, f)| f.online.iter().map(move |&k| (i, k)))
            .collect()
    }

    /// `[x | BoW(text)]` and, per tokenized feature, the token ids of every
    /// row.
    pub(crate) fn apply(
        &self,
        x: ArrayView2<f64>,
        text: &[Vec<String>],
    ) -> Result<(Array2<f64>, Vec<Vec<TokenCounts>>)> {
        if text.len() != self.n_columns || text.iter().any(|c| c.len() != x.nrows()) {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} text columns of {} rows",
                self.n_columns,
                x.nrows()
            )));
        }
        let tokenized: Vec<Vec<Vec<String>>> = text
            .iter()
            .map(|col| col.iter().map(|t| self.tokenize(t)).collect())
            .collect();
        let mut out = Array2::zeros((x.nrows(), x.ncols() + self.n_bow()));
        out.slice_mut(ndarray::s![.., ..x.ncols()]).assign(&x);
        let mut offset = x.ncols();
        let mut ids = Vec::with_capacity(self.features.len());
        for f in &self.features {
            let index: HashMap<&str, u32> = f
                .tokens
                .iter()
                .enumerate()
                .map(|(i, t)| (t.as_str(), i as u32))
                .collect();
            let rows: Vec<TokenCounts> = tokenized[f.column]
                .iter()
                .map(|tokens| token_counts(tokens, f.gram_order, &index))
                .collect();
            for &top in &f.bow {
                for (r, row) in rows.iter().enumerate() {
                    for &(id, _) in row {
                        if (id as usize) < top {
                            out[[r, offset + id as usize]] = 1.0;
                        }
                    }
                }
                offset += top;
            }
            ids.push(rows);
        }
        Ok((out, ids))
    }
}

/// The grams of `tokens` (`gram_order` consecutive words joined by `\0`).
fn grams(tokens: &[String], gram_order: usize) -> impl Iterator<Item = String> + '_ {
    tokens.windows(gram_order).map(|w| w.join("\0"))
}

/// The dictionary's tokens in id order: counted on the rows, at least
/// `occurrence_lower_bound` times, by count descending then by the words.
fn build_dictionary(rows: &[Vec<String>], d: &CatBoostTextDictionary) -> Vec<String> {
    let mut counts: HashMap<String, u64> = HashMap::new();
    for tokens in rows {
        for g in grams(tokens, d.gram_order) {
            *counts.entry(g).or_default() += 1;
        }
    }
    let mut kept: Vec<(Vec<&str>, u64, &String)> = counts
        .iter()
        .filter(|&(_, &c)| c >= d.occurrence_lower_bound)
        .map(|(g, &c)| (g.split('\0').collect(), c, g))
        .collect();
    // Grams compare word by word (`CompareNGram`).
    kept.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    kept.truncate(d.max_dictionary_size.unwrap_or(usize::MAX));
    kept.into_iter().map(|(_, _, g)| g.clone()).collect()
}

/// The known grams of a text with counts, sorted by id.
fn token_counts(tokens: &[String], gram_order: usize, index: &HashMap<&str, u32>) -> TokenCounts {
    let mut ids: Vec<u32> = grams(tokens, gram_order)
        .filter_map(|g| index.get(g.as_str()).copied())
        .collect();
    ids.sort_unstable();
    let mut out: TokenCounts = Vec::new();
    for id in ids {
        match out.last_mut() {
            Some((last, n)) if *last == id => *n += 1,
            _ => out.push((id, 1)),
        }
    }
    out
}

/// An online calcer's state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) enum OnlineCalcer {
    NaiveBayes(NaiveBayes),
    Bm25(Bm25),
}

impl OnlineCalcer {
    pub(crate) fn new(kind: OnlineKind, n_classes: usize) -> Self {
        match kind {
            OnlineKind::NaiveBayes => Self::NaiveBayes(NaiveBayes::new(n_classes)),
            OnlineKind::Bm25 => Self::Bm25(Bm25::new(n_classes)),
        }
    }

    /// Features per text.
    pub(crate) fn n_features(kind: OnlineKind, n_classes: usize) -> usize {
        match kind {
            OnlineKind::NaiveBayes => NaiveBayes::n_features(n_classes),
            OnlineKind::Bm25 => n_classes,
        }
    }

    pub(crate) fn compute(&self, text: &[(u32, u32)]) -> Vec<f32> {
        match self {
            Self::NaiveBayes(c) => c.compute(text),
            Self::Bm25(c) => c.compute(text),
        }
    }

    pub(crate) fn update(&mut self, class: usize, text: &[(u32, u32)]) {
        match self {
            Self::NaiveBayes(c) => c.update(class, text),
            Self::Bm25(c) => c.update(class, text),
        }
    }
}

/// CatBoost's `TMultinomialNaiveBayes` (priors 0.5, a seen-token prior of
/// 1): class frequencies of tokens, updated row by row
/// (`TNaiveBayesVisitor`).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct NaiveBayes {
    docs: Vec<f64>,
    total: Vec<f64>,
    freq: Vec<HashMap<u32, u64>>,
    seen: SeenTokens,
}

/// The visitor's `TDenseHashSet<ui32>` of seen tokens, whose size is the
/// `NumSeenTokens` prior. Its empty marker is 0, the id of the most frequent
/// token: inserting token 0 always finds an "empty" bucket and counts once
/// more, until a rehash (on reaching the grow threshold) drops it again.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SeenTokens {
    keys: HashSet<u32>,
    filled: usize,
    buckets: usize,
    threshold: usize,
}

impl Default for SeenTokens {
    fn default() -> Self {
        // 1 << LogInitSize (8) buckets, MaxLoadFactor 50.
        Self {
            keys: Default::default(),
            filled: 0,
            buckets: 256,
            threshold: 127,
        }
    }
}

impl SeenTokens {
    fn insert(&mut self, token: u32) {
        if token != 0 && !self.keys.insert(token) {
            return;
        }
        self.filled += 1;
        if self.filled >= self.threshold {
            self.buckets *= 2;
            self.threshold = (self.buckets / 2).max(1) - 1;
            self.filled = self.keys.len();
        }
    }
}

impl NaiveBayes {
    fn new(n_classes: usize) -> Self {
        Self {
            docs: vec![0.0; n_classes],
            total: vec![0.0; n_classes],
            freq: vec![HashMap::new(); n_classes],
            seen: SeenTokens::default(),
        }
    }

    /// Features per text: one for two classes (the first class's
    /// probability), else one per class.
    fn n_features(n_classes: usize) -> usize {
        if n_classes > 2 {
            n_classes
        } else {
            1
        }
    }

    /// `Compute`: the softmax of the classes' log probabilities (float32).
    fn compute(&self, text: &[(u32, u32)]) -> Vec<f32> {
        let k = self.docs.len();
        let mut lp: Vec<f64> = (0..k)
            .map(|c| {
                let mut value = (self.docs[c] + 0.5).ln();
                let mut class_tokens = self.total[c] + 0.5 * (self.seen.filled as f64 + 1.0);
                let mut len = 0.0;
                for &(t, n) in text {
                    len += f64::from(n);
                    let num = match self.freq[c].get(&t) {
                        Some(&f) => 0.5 + f as f64,
                        None => {
                            class_tokens += 0.5;
                            0.5
                        }
                    };
                    value += f64::from(n) * num.ln();
                }
                value - len * class_tokens.ln()
            })
            .collect();
        let max = lp.iter().copied().fold(lp[0], f64::max);
        let mut total = 0.0;
        for v in &mut lp {
            *v = (*v - max).exp();
            total += *v;
        }
        lp.iter_mut().for_each(|v| *v /= total);
        lp[..Self::n_features(k)]
            .iter()
            .map(|&v| v as f32)
            .collect()
    }

    fn update(&mut self, class: usize, text: &[(u32, u32)]) {
        for &(t, n) in text {
            self.seen.insert(t);
            *self.freq[class].entry(t).or_default() += u64::from(n);
            self.total[class] += f64::from(n);
        }
        self.docs[class] += 1.0;
    }
}

/// CatBoost's `TBM25` (k 1.5, b 0.75, the inverse class frequency
/// truncated at 1e-3): per class, the BM25 score of the text against the
/// class's tokens.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct Bm25 {
    /// All tokens seen, from 1.
    total: u64,
    class_total: Vec<u64>,
    freq: Vec<HashMap<u32, u64>>,
}

impl Bm25 {
    const K: f64 = 1.5;
    const B: f64 = 0.75;
    const TRUNCATE: f64 = 1e-3;

    fn new(n_classes: usize) -> Self {
        Self {
            total: 1,
            class_total: vec![0; n_classes],
            freq: vec![HashMap::new(); n_classes],
        }
    }

    fn compute(&self, text: &[(u32, u32)]) -> Vec<f32> {
        let k = self.freq.len();
        let mean = self.total as f64 / k as f64;
        let mut scores = vec![0.0f64; k];
        let mut tf = vec![0u64; k];
        for &(t, _) in text {
            let mut with_term = 0;
            for (c, freq) in self.freq.iter().enumerate() {
                tf[c] = freq.get(&t).copied().unwrap_or(0);
                with_term += usize::from(tf[c] > 0);
            }
            let idf = (((k - with_term) as f64 + 0.5) / (with_term as f64 + 0.5))
                .ln()
                .max(Self::TRUNCATE);
            for c in 0..k {
                if tf[c] == 0 {
                    continue;
                }
                let f = tf[c] as f64;
                scores[c] += idf * f * (Self::K + 1.0)
                    / (f + Self::K * (1.0 - Self::B + Self::B * mean / self.class_total[c] as f64));
            }
        }
        scores.into_iter().map(|v| v as f32).collect()
    }

    fn update(&mut self, class: usize, text: &[(u32, u32)]) {
        for &(t, n) in text {
            *self.freq[class].entry(t).or_default() += u64::from(n);
            self.class_total[class] += u64::from(n);
            self.total += u64::from(n);
        }
    }
}
