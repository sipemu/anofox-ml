# Changelog

All notable changes to anofox-ml are recorded here. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Registry

- `DynModel::predict_log_proba` (#125) for GaussianNB, MultinomialNB,
  BernoulliNB, ComplementNB, CategoricalNB (their joint log-likelihoods,
  finite where `predict_proba` underflows), LinearDiscriminantAnalysis,
  QuadraticDiscriminantAnalysis, LogisticRegressor, Svc and GaussianMixture.
- `DynModel::kneighbors(x, k)` (#125) for KnnClassifier and KnnRegressor:
  distances and training-row indices, `n × k` each, nearest first
  (`k = None`: the model's `n_neighbors`).
- `DynModel::partial_fit(x, y, sample_weight, classes)` (#125) for the five
  naive Bayes kinds, StandardScaler, MinMaxScaler, MaxAbsScaler,
  IncrementalPca (batches give the same model as one fit), SgdClassifier,
  SgdRegressor, PassiveAggressiveRegressor, MlpClassifier and MlpRegressor
  (one epoch per batch). A failed batch leaves the model unchanged.
- `warm_start` on `DynModel` for MlpClassifier and MlpRegressor (#125):
  up to `max_iter` further epochs from the fitted weights.
- `DynModel::fit_with_offset(x, y, sample_weight, offset)` and
  `predict_with_offset(x, offset)` (#125) for PoissonRegressor,
  GammaRegressor and TweedieRegressor: an offset on the link scale, e.g.
  `ln(exposure)`; `predict` predicts at offset 0.
- `GammaRegressor::fit_weighted_with_offset`.
- New registry kinds (#125): `LassoLarsIC` (AIC / BIC), `PlsRegression`
  (predict and transform), both matching scikit-learn 1.8, `IsotonicRegressor`
  and `BinomialRegressor` (with offsets; weights as trial counts).
  `PlsRegression` and `IsotonicRegressor` configs are now `Serialize`.
- `GaussianProcessRegressor` (with `DynModel::predict_std`) and
  `GaussianProcessClassifier` (binary, or one-vs-rest) in the registry
  (#125), defaulting to scikit-learn's `1.0 * RBF(1.0)` with fixed
  hyperparameters. The Gaussian process configs, kernels and fitted models
  are now `Serialize` (the Cholesky factors are stored, so a loaded model is
  bit-identical).
- `Cca` in the registry (#125), fitted on `[X | Y]` with `y_columns` (the
  last columns are `Y`); `transform` gives the X scores and the new
  `DynModel::transform_y` the Y scores, matching scikit-learn 1.8. `Cca` and
  `FittedCca` are now `Serialize`.
- `BaggingEstimatorRegressor` / `BaggingEstimatorClassifier` in the
  registry (#125): bagging over any registered kind via a nested
  `{estimator, estimator_params}` (sklearn's `Bagging*(estimator=...)`),
  with the tree bagging kinds' sampling settings. The existing
  `BaggingRegressor` / `BaggingClassifier` kinds are unchanged.
- `VotingRegressor` / `VotingClassifier` in the registry (#125), over a
  list of `{estimator, estimator_params}` with optional `weights` and
  (classifier) `voting: "hard" | "soft"`, matching scikit-learn 1.8.

## [0.4.0] - 2026-11-06

Gradient boosting (#127): HistGradientBoosting parity with scikit-learn 1.8,
LightGBM 4.7 parity of the LGBM engine (with DART and EFB), faster boosting,
a wider registry surface, and the CatBoost-style estimators. These match
catboost 1.2.10 down to its random generator and default to its defaults
(see the breaking note under bootstrap).

### CatBoost-style boosting: bootstrap (#127 §5)

- `bootstrap_type` (CatBoost's defaults: MVS, or Bayesian for MultiClass),
  `subsample` (0.8 for MVS, 0.66 for Bernoulli, 1 below 100 rows),
  `bagging_temperature` and `mvs_reg` on `CatBoostRegressor` /
  `CatBoostClassifier`. Bootstrap is drawn per tree, as CatBoost does it:
  - **Bayesian:** weights `powf(-FastLogf(u), t)` in blocks of 1000 rows.
  - **MVS** (`TMvsSampler`): λ from the last tree's leaves (the first tree:
    from the mean gradient in 128 blocks), the threshold found by libc++'s
    `std::partition` recursion per block of 8192 rows, rows kept with
    probability `min(1, g / threshold)` and weight `1 / p`.
  - **Bernoulli:** one draw per row.
- The weights only reach the scores (`CalcWeightedData`, float32); the leaf
  values use the sample weights as before.
- **Breaking:** with all of this (#144–#149) the CatBoost-style estimators now
  default to CatBoost's own defaults: `has_time=false`,
  `random_strength=1`, MVS bootstrap, CatBoost's `max_ctr_complexity` and
  `leaf_estimation_iterations`. Pass `has_time=true`, `random_strength=0`
  and `bootstrap_type=No` for the deterministic behaviour of earlier
  versions.
- Matches catboost 1.2.10 to 7e-16 on 12 golden cases. A 20k-row fit with
  **all CatBoost defaults** (300 trees; MVS, score noise, permutations, CTR
  combinations, 10 leaf steps) matches to 1.3e-15, plain and ordered.

### CatBoost-style boosting: score noise (#127 §5)

- `random_strength` (CatBoost's default 1, `NormalWithModelSizeDecrease`) on
  `CatBoostRegressor` / `CatBoostClassifier`. The noise scale is the root
  mean square of the weighted derivatives times `m / (1 + m)`, with
  `m = exp(ln n − iteration · lr)`. Every split's score gets normal noise
  from a generator per candidate entry (`SetBestScore`: seeded with the
  level's seed plus the entry's index, advanced by 10), and every
  candidate's best score gets noise again from the main generator
  (`SelectBestCandidate`, comparing gains over the previous level's score).
- To get the seeds right, the candidate list now follows CatBoost's layout:
  non-binary float and one-hot features, packs of eight binary features
  (one-border floats and two-valued categoricals), simple CTRs per
  feature, then tree CTRs per projection. The projections are visited in
  the iteration order of Yandex's `THashSet`, emulated with
  `TProjection::GetHash`, prime bucket counts and head insertion.
  Projections keep their conditions in tree order, as CatBoost's do.
- Matches catboost 1.2.10 to 7e-16 on 12 golden cases (RMSE, Logloss and
  MultiClass, plain and ordered, CTRs and combinations, seeds, 10 leaf
  steps). A 20k-row fit with CatBoost's defaults except bootstrap matches to
  1.1e-15.

### CatBoost-style boosting: permutations (#127 §5)

- `has_time` (now defaulting to false, as in CatBoost), `random_seed`
  (default 0) and `permutation_count` (default 4) on `CatBoostRegressor` /
  `CatBoostClassifier`. CatBoost's random numbers are reproduced: the
  `TFastRng64` generator (checked against Yandex `util`'s test vectors),
  with its uniform draws by rejection, `GenRandReal1`, the polar normal and
  Fisher-Yates over blocks of `min(256, n / 1000 + 1)` rows. They are drawn
  in CatBoost's order:
  - the initial shuffle of the rows (`ShuffleLearnDataIfNeeded`: with
    categorical features or ordered boosting);
  - the learning folds' permutations (`permutation_count - 1` folds, the
    first one unshuffled), then the averaging fold's (with CTRs);
  - per tree the fold to search on and a seed. Per level: `rsm` draws per
    feature, simple CTR feature and tree-CTR base, a seed, and a score
    instance per candidate, with binary features packed eight to a
    candidate. Then a seed, and one draw per leaf-estimation step.
- Each fold has its own online CTRs, approximations and body / tail prefixes.
  Every fold is moved by every tree, and plain boosting shares fold 0 with an
  unpermuted averaging fold (`UseAveragingFoldAsFoldZero`).
- Matches catboost 1.2.10 (`random_strength=0`, `bootstrap_type='No'`) to
  6e-16 on 13 golden cases: plain and ordered; RMSE, Logloss and MultiClass;
  CTRs and combinations; weights; seed 7; 10 leaf steps; 1200 rows. On
  20k rows it matches RMSE to 5e-15; Logloss can rarely break a near-tie
  differently (see the module docs).

### CatBoost-style boosting: leaf estimation iterations (#127 §5)

- `leaf_estimation_iterations` and `leaf_estimation_backtracking`
  (`AnyImprovement` / `No`) on `CatBoostRegressor` / `CatBoostClassifier`.
  This is CatBoost's `FastGradientWalker`: several Newton or Gradient steps
  per leaf estimation, each from the derivatives at the moved
  approximations. With backtracking a step is halved until the objective
  improves, and every try uses up an iteration. The defaults are CatBoost's:
  Logloss takes 10 Newton / 40 Gradient steps and MultiClass 1 Newton /
  10 Gradient, but every loss takes 1 below 200 iterations with fewer than
  20 features. Previously every fit took a single step.
- The backtracking objective is computed as CatBoost computes it.
  - The model's leaves are evaluated at the moved point plus the step, and
    the learning folds at the base plus the summed steps.
  - Logloss uses CatBoost's float32 `VectorFastLogf` metric on `exp(a)`,
    summed in four lanes per block of 4096 rows.
  - Ordered Logloss folds move by `FastLogf(exp(δ)) · lr`
    (`ApplyLearningRate`).
- Matches catboost 1.2.10 (`thread_count=1`) to 8e-16 on 12 golden cases.
  A 20k-row check with CatBoost's full defaults (CTR combinations, 10 leaf
  steps) matches to 1.2e-15, plain and ordered.

### CatBoost-style boosting: CTR feature combinations (#127 §5)

- `max_ctr_complexity` on `CatBoostRegressor` / `CatBoostClassifier`
  (CatBoost's default: 4, or 1 below 200 iterations): CTRs of feature
  combinations, as CatBoost's `AddTreeCtrs` builds them per tree level. The
  conditions of the tree's float and one-hot splits, and the projection of
  each of its CTR splits, are extended by every categorical feature with
  CTRs. The model-size penalty takes its maximum over the level's CTR
  candidates. A projection's CTR columns are cached across trees, and a
  fitted model keeps only the CTRs its splits use. Matches catboost 1.2.10
  to 6e-16 on 8 golden cases. They include cat × cat, cat × float
  conditions (up to 3), cat × one-hot, multiclass, ordered and weighted
  fits.

### CatBoost-style boosting: multiclass (#127 §5)

- `CatBoostClassifier` fits more than two classes with CatBoost's
  `MultiClass` loss: one approximation per class, softmax derivatives with
  the packed Hessian, and the Newton step `(−H + λI)⁻¹ g` through the
  reference LAPACK `dppsv` (packed Cholesky) with CatBoost's float32 L2
  scaling. The split scores add up over the classes. Ordered boosting
  updates the tail rows one at a time, as CatBoost's
  `CalcApproxDeltaMulti` does. Categorical features get one `Borders` CTR
  per target border (`classes − 1`) and prior. `raw_predictions` returns
  the class scores; the registry's `decision_function` returns them for
  multiclass models (n × classes) and the log-odds for binary ones.
  `boost_from_average` is rejected for multiclass, as in CatBoost.
  Matches catboost 1.2.10 to 6e-16 on 15 golden cases (Cosine / L2,
  Newton / Gradient, weighted, 5 classes, arbitrary labels, ordered,
  CTRs).

### CatBoost-style boosting: regressor CTRs (#127 §5)

- `CatBoostRegressor` now builds the `Borders` CTRs of its categorical
  features too, on CatBoost's single MinEntropy target border
  (`SelectBorders`: the most balanced split of the distinct targets, at
  the float32 midpoint). Matches catboost 1.2.10 to 4e-16 (plain, ordered,
  weighted).

### CatBoost-style boosting: ordered boosting (#127 §5)

- `boosting_type = Ordered` (and `fold_len_multiplier`, default 2) on
  `CatBoostRegressor` / `CatBoostClassifier`: CatBoost's ordered boosting
  against prediction shift. The rows (in the given order, as
  `has_time=True`) form body / tail prefixes (`SelectMinBatchSize`, then
  `ceil(body · multiplier)`), each with its own approximation that only its
  body rows move; splits are scored by body-row leaf estimates on the tail
  rows with per-prefix L2 scaling (`CalcStatsKernel`,
  `UpdateScoreBinKernelOrdered`); the model's leaf values come from all
  rows. Matches catboost 1.2.10 to 8e-16 on 7 golden cases (incl. 1200
  rows, weights, Gradient leaves and categorical CTRs). Python:
  `boosting_type="Ordered"`, `fold_len_multiplier=`.

### CatBoost-style boosting: categorical features (#127 §5)

- `cat_features` on `CatBoostClassifier`: categorical features with at
  most `one_hot_max_size` (2) categories are split one-hot, the others
  through CatBoost's ordered target statistics, its default CTRs
  `Borders` (priors 0, 0.5, 1) and `Counter`: online counts of the rows
  before each row on the training data, the final counts for predictions,
  quantized into 15ths as CatBoost's `CalcCTR`, with the `model_size_reg`
  penalty (0.5) on CTRs new to the model. Matches catboost 1.2.10 to 3e-16
  (`max_ctr_complexity=1`; feature combinations are not supported).
  Python: `cat_features` with any category values (unseen ones allowed).
  (`CatBoostRegressor`: see the regressor CTRs entry.)

### CatBoost-style boosting: oblivious trees (#127 §5)

- `CatBoostRegressor` (RMSE) and `CatBoostClassifier` (Logloss, two
  classes) on oblivious (symmetric) trees, with CatBoost's parameter names
  and defaults (`iterations`, `learning_rate`, `depth`, `l2_leaf_reg`,
  `border_count`, `score_function` Cosine / L2, `leaf_estimation_method`
  Newton / Gradient, `boost_from_average`). The numerics are CatBoost
  1.2's: GreedyLogSum quantization (with libc++'s heap order, which decides
  its ties), float32 features / targets / learning rate, `nan_mode=Min`,
  the Cosine score in its SSE summation order, the scaled L2
  regularization, redundant-split removal, plain boosting. With CatBoost's
  randomness off (`random_strength=0`, no bootstrap, `has_time=True`) the
  models match catboost 1.2.10 to 1e-15 on 14 golden cases and its borders
  exactly. Registry kinds and Python `CatBoostRegressor` /
  `CatBoostClassifier`. Not yet: categorical features (ordered target
  statistics), ordered boosting, multiclass, the bootstrap and score
  noise.

### Registry / Python: boosting surface (#127 §4, #125)

- `DynModel` gains `feature_importances()` (decision trees, random forests,
  extra trees, gradient boosting, bagging, HistGradientBoosting, LGBM),
  `staged_predict(x)` (an `n × stages` array; gradient boosting,
  HistGradientBoosting, LGBM), `set_params(json)` (merges overrides and keeps
  the fitted state), `warm_start(x, y, w)` (gradient boosting: adds stages up
  to `n_estimators`, unchanged on error) and `fit_with_validation(...)`
  (HistGradientBoosting's explicit validation set); each is listed in
  `capabilities()`. `ExtraTreesClassifier` now lists `predict_proba`.
- Continued training for the LightGBM engine: `LgbmRegressor` /
  `LgbmClassifier::fit_warm_start(&prev, x, y, opts)` and
  `HistGradientBoosting*::fit_warm_start(&prev, x, y, w)` keep the fitted
  bins and trees, rebuild the training scores and replay the bagging draws,
  so on the same data the result equals a single fit with `n_estimators`
  stages (tested bit for bit). GOSS, DART, early stopping, `init_score`,
  `max_features < 1` and a changed learning rate are errors. Registry
  `warm_start` for these kinds; Python `warm_start=` on
  `HistGradientBoosting*`, matching sklearn 1.8's warm start to 1e-12.
- `FittedGradientBoostingRegressor::staged_predict` and
  `FittedGradientBoostingClassifier::staged_decision_function` /
  `staged_predict_proba` / `staged_predict` (the last stage equals the
  final prediction exactly).
- Python: `feature_importances_` (only on estimators that have it),
  `staged_predict` on the gradient boosting, HistGradientBoosting and LGBM
  estimators, `warm_start=` on `GradientBoosting*`, and
  `HistGradientBoosting*.fit(..., X_val=, y_val=, sample_weight_val=)`, which
  matches sklearn 1.8 to 1e-12.

### LGBM / HistGradientBoosting: speed (#127)

- Profiled on 100k × 50 and rebuilt the hot paths. The models stay bit for
  bit the same as before (the sklearn and LightGBM golden suites are
  unchanged), except that nodes above 16 384 rows in LightGBM mode get
  children's totals summed in fixed chunks (rounding-level, independent of
  the thread count):
  - split search: a candidate's outputs only for a new best, counts
    estimated inline instead of cloning each histogram, features scanned in
    parallel and reduced in order;
  - histograms: the node's gradients gathered once as float32 pairs (GOSS
    now rounds its amplified gradients to float32, as LightGBM), `u32` rows,
    bin sums indexed by the `u8` bin, counts only in sklearn mode,
    sequential columns at a root holding every row;
  - row partitions fused with the children's totals and parallel for large
    nodes; subtraction without a clone; binning, training-score updates and
    prediction parallel.
- `n_jobs` on `LgbmRegressor` / `LgbmClassifier` (Python `n_jobs=`):
  `None` fits on the physical cores, which beats all logical threads on
  SMT / hybrid CPUs. Models do not depend on it.
- `benchmarks/boosting_benchmark.py`: fit / predict against sklearn and
  LightGBM, results in the README.

### LGBM engine: exclusive feature bundling (#127)

- LightGBM's EFB (`enable_bundle`, on by default; `with_enable_bundle` on
  `LgbmRegressor` / `LgbmClassifier`, Python `enable_bundle=`): features
  that are never off their most frequent bin in the same row (one-hot
  blocks, sparse columns) share one bin column, found greedily in feature
  order and by non-default count as LightGBM does. A node takes one pass
  over its rows per bundle, and each member's most frequent bin comes from
  the node totals (LightGBM's `FixHistogram`). Only exactly exclusive
  features are bundled, so the model is unchanged up to rounding (LightGBM
  also allows `n / 10000` conflicts). Off in the HistGradientBoosting
  facade (sklearn has no bundling).
- Histograms are sized to each feature's bins instead of 255. Together:
  50 000 × 310 one-hot-heavy data trains about 2× faster.
- Weighted L2 rounds the residual to float32 before weighting, as LightGBM:
  weighted regression now matches LightGBM 4.7.0 to 2e-16 (was 5e-9).
- 5 golden cases on one-hot / sparse data (with and without bundling) in
  `lgbm_parity.json`.

### LGBM engine: DART (#127)

- `BoostingType::Dart` with `DartParams` (`drop_rate`, `max_drop`,
  `skip_drop`, `uniform_drop`, `xgboost_dart_mode`, `drop_seed`; LightGBM's
  names and defaults) via `with_dart(...)` on `LgbmRegressor` /
  `LgbmClassifier`: LightGBM's `dart.hpp` with its LCG for the dropping,
  weight-proportional or uniform drops, both normalisations, and the
  initial score carried by the first tree. Matches LightGBM 4.7.0 bit for
  bit (regression objectives incl. L1 refits, binary, multiclass). Not
  combined with early stopping or `init_score` (as in LightGBM, where DART
  has no early stopping). Python: `boosting_type="dart"` and the DART
  parameters on `LGBMRegressor` / `LGBMClassifier`.

### LGBM engine: LightGBM 4.7 parity (#127)

- `LgbmRegressor` / `LgbmClassifier` now reproduce LightGBM 4.7.0 to
  rounding (most cases bit for bit) for every objective, with weights,
  NaNs, categorical features, monotone constraints, L1 / L2 penalties and
  the default `min_child_samples=20`:
  - binning is LightGBM's `FindBin`: zero as its own bin
    (`FindBinWithZeroAsOneBin`, the `±1e-35f` bounds), `GreedyFindBin`,
    `nextafter` bounds, a missing-value bin taken from the budget;
  - split search counts are estimated from the hessians
    (`round(hess · n / (H + 2 kEpsilon))`), the right child is scanned from
    the top bin down (missing values left), then, for features with missing
    values, the left child from the bottom up; `kEpsilon` in the hessian
    sums; the gain shift is the unconstrained node gain; `min_split_gain`
    is compared with LightGBM's (unhalved) gain;
  - float32 labels, weights, gradients and quantile level; LightGBM's
    `PercentileFun` for the L1 / quantile refits;
  - missing values in features without any in training are treated as 0.
  LGBM fits change accordingly (closing the hessian-count gap noted with
  the Poisson / Gamma objectives).
- New generator `gen_lgbm_parity.py` (pinned `lightgbm==4.7.0`) and
  `golden_lgbm_parity.rs` (21 cases, tolerance 1e-8).

### HistGradientBoosting: early-stopping scorers and explicit validation sets (#127)

- `with_scoring(EarlyStoppingScoring::{Loss, Default, R2,
  NegMeanSquaredError, NegMeanAbsoluteError, Accuracy, NegLogLoss})`
  (sklearn's `scoring`; `Default` is the estimator's `score`). As in
  sklearn, scorers see the predictions of the raw scores and the training
  scores use at most 10 000 training rows (`resample`, stratified for the
  classifier, with the split's seed). Python: `scoring=None` or one of
  these names.
- `fit_with_validation(x, y, w, x_val, y_val, w_val)` on both facades
  (sklearn's `fit(..., X_val=, y_val=, sample_weight_val=)`): early stopping
  on the given rows, without a split. (Python support follows with the
  registry work in #127 §4.)
- `anofox_ml_core::numpy_random::resample_without_replacement` (sklearn's
  `resample(..., replace=False[, stratify])`).

### Staged predictions for HistGradientBoosting and LGBM (#127)

- `staged_predict` on the fitted HistGradientBoosting regressor /
  classifier and the LGBM regressor / classifier, and
  `staged_predict_proba` on the classifiers (sklearn's methods of the same
  names): lazy iterators with one prediction per boosting iteration,
  the last equal to `predict` / `predict_proba`. Match scikit-learn 1.8.0
  at every stage. (Python and registry support follows with #127 §4.)

### HistGradientBoosting: max_features and class_weight (#127)

- `with_max_features(f)` (sklearn's `max_features`, in (0, 1]): every split
  tries a random `ceil(f · n_allowed)` of the features. The subsets are
  sklearn's for the same `random_state` (its second seed feeds numpy's
  `default_rng`), and as in sklearn the nodes it cannot split are not
  evaluated, so the draws stay aligned across trees.
- `HistGradientBoostingClassifier::with_class_weight` (`Balanced` or
  per-label weights), multiplied into the sample weights before the
  early-stopping split, as sklearn does. Python: `max_features`,
  `class_weight` (dict keys are the original labels; sklearn 1.8 looks
  them up among its encoded labels instead).
- `anofox_ml_core::numpy_random::Generator`: numpy's `default_rng` (PCG64
  seeded through `SeedSequence`) with `shuffle`.

### HistGradientBoosting: early stopping (#127)

- `with_early_stopping(EarlyStopping::{Auto, Enabled, Disabled})`
  (sklearn's `early_stopping`; `Auto`, the default, is on above 10 000
  samples), `with_validation_fraction(Option<TestSize>)` (fraction, count,
  or `None` for the training loss), `with_n_iter_no_change`, `with_tol`,
  `with_random_state`. The validation rows are split off exactly as
  sklearn's `train_test_split` does (stratified for the classifier), the
  scores are sklearn's negative mean losses, and sklearn's stopping rule
  keeps every tree. Fitted models report `n_iter()`, `train_score()` and
  `validation_score()` (sklearn's `n_iter_`, `train_score_`,
  `validation_score_`). Python: the same parameters (`scoring="loss"`
  only) and attributes. Matches scikit-learn 1.8.0 to rounding, iteration
  counts and score curves included.
- `anofox_ml_core::numpy_random`: numpy's legacy `RandomState` (MT19937,
  `randint`, `permutation`, `shuffle`, `choice`) and sklearn's
  `train_test_split` / stratified split on top of it, draw for draw.
- Fix: sklearn binning on more than `max_bins` distinct values now
  computes the `midpoint` percentiles in numpy's arithmetic. Thresholds
  whose index lands exactly on a data point could differ from sklearn by
  one value.
- `Auto` early stopping changes HistGradientBoosting fits on more than
  10 000 samples (as in sklearn); use `EarlyStopping::Disabled` for the
  previous behaviour.

### HistGradientBoosting: monotonic and interaction constraints (#127)

- `with_monotonic_cst(vec)` (sklearn's `monotonic_cst`; `1` / `-1` / `0`
  per feature) on both facades, validated as sklearn does (length,
  values, not on categorical features, not for multiclass classifiers).
  sklearn's node bounds and gains: child outputs clipped to
  `[lower, mid]` / `[mid, upper]`, gains from the node's stored value.
- `with_interaction_cst(InteractionConstraints::{Groups, Pairwise,
  NoInteractions})` (sklearn's `interaction_cst`): a branch only splits on
  features of the groups that contain every feature split on above it;
  unlisted features form one more group. Indices always refer to the
  input columns (sklearn reads them after moving categorical columns
  first).
- Python: `monotonic_cst` and `interaction_cst`. Matches scikit-learn
  1.8.0 to rounding for regressors and binary / multiclass classifiers
  (`gen_hist_gb_parity.py`).

### HistGradientBoosting: categorical features (#127)

- `with_categorical_features(indices)` on both HistGradientBoosting
  facades (sklearn's `categorical_features`; Python: indices or a boolean
  mask). Each distinct value is a category (at most `max_bins`, otherwise
  an error); `NaN` and categories unseen in training are treated as
  missing. Splits follow sklearn: categories with enough support sorted by
  `G / (H + 10)`, scanned from both ends to the middle, missing values as a
  category of their own, categorical features ahead of numeric ones in
  ties. Matches scikit-learn 1.8.0 to rounding for regression losses and
  binary / multiclass classification (`gen_hist_gb_parity.py`).
- `BinningMethod::Sklearn` now gives every category its own bin (the
  first `max_bins` in sorted order) instead of LightGBM's rare-category
  "other" bin.

### HistGradientBoosting: Poisson and Gamma losses, exact sklearn parity (#127)

- `HistGradientBoostingLoss::{Poisson, Gamma}` (sklearn's `loss="poisson"` /
  `"gamma"`, log link; predictions are `exp` of the raw score). The engine
  gains `LgbmObjective::{Poisson, Gamma}` with LightGBM's
  `poisson_max_delta_step` (default 0.7; the facade uses sklearn's exact
  hessian). Targets are validated (Poisson: non-negative with a positive
  sum; Gamma: positive). Python: `loss="poisson"` / `"gamma"`.
- The HistGradientBoosting facades now reproduce sklearn bit for bit (to
  ~1e-16) for every regression loss, weighted or not: float32 gradients
  and hessians, sklearn's split scan order and gain arithmetic (ties
  between mirrored splits are common with the L1 / quantile losses),
  sklearn's routing of missing values unseen in training, its weighted
  percentile (`inverted_cdf`) and numpy's interpolation for the leaf
  refits. Previously absolute error and quantile drifted from sklearn by
  10–30 % after a dozen iterations. Fits made with these losses change
  accordingly; LightGBM-mode (`Lgbm*`) fits are unchanged.
- New golden generator `gen_hist_gb_parity.py` (all five losses, with and
  without sample weights, NaNs in a feature); the engine's Poisson / Gamma
  objectives are checked against LightGBM 4.6.
- Known gap (LightGBM mode): LightGBM estimates leaf counts from hessians
  for `min_child_samples`; the engine counts rows, so objectives with
  uneven hessians (Gamma) can split differently from LightGBM.

### New estimator: KNeighborsTransformer (#113)

- `KNeighborsTransformer` in `anofox-ml-neighbors` (distance graph with
  `n_neighbors + 1` neighbours, or connectivity), returned dense. Registry
  kind and Python class; matches scikit-learn 1.8.0
  (`gen_kneighbors_transformer.py`).

### New estimator: SelfTrainingClassifier (#113, batch 8)

- Registry kind `SelfTrainingClassifier` over any registered classifier
  with `predict_proba` (unlabelled rows `−1`): pseudo-labels by a
  probability threshold or the k most confident rows per round, up to
  `max_iter` rounds, then a refit; `transduction`, `labeled_iter`,
  `n_iter` and the termination reason. Python class in
  `anofox_ml_python.multiclass`. Matches scikit-learn 1.8.0 over GaussianNB
  (`gen_self_training.py`).

### New estimator: FeatureAgglomeration (#113, batch 8)

- `FeatureAgglomeration` in `anofox-ml-cluster`: the columns are clustered
  by `AgglomerativeClustering` (ward / complete / average / single) and
  each cluster pooled by its mean; `labels`, `transform`,
  `inverse_transform`. Registry kind and Python class; matches
  scikit-learn 1.8.0 exactly (`gen_feature_agglomeration.py`).

### New estimators: semi-supervised label propagation (#113, batch 8)

- `LabelPropagation` and `LabelSpreading` in `anofox-ml-neighbors`
  (unlabelled rows have the label `−1`): RBF or k-nearest-neighbour
  affinity graphs, propagation with clamped labelled rows, spreading with
  scipy's normalised Laplacian and `alpha`; `label_distributions`,
  `transduction`, `predict` and `predict_proba`. Registry kinds and Python
  classes; match scikit-learn 1.8.0 exactly (`gen_semi_supervised.py`:
  iterations, distributions to 1e-10, transduction and predictions).

### New estimators: cross-validated linear models (#113, batch 7)

- `LogisticRegressionCV` in `anofox-ml-regression`: `C` from a grid
  (`CGrid::Count(n)` = `logspace(−4, 4, n)`, or explicit values) by
  unshuffled stratified `cv`-fold CV, scored by accuracy (sklearn's
  default) or negative log loss, summed over the folds, refit on all rows;
  every other setting comes from the wrapped `LogisticRegressor`.
- `OrthogonalMatchingPursuitCV`: the sparsity with the lowest mean
  squared error along the OMP path `1..max_iter` (unshuffled `cv`-fold),
  refit at that size.
- Registry kinds and Python classes; both match scikit-learn 1.8.0
  (`gen_cv_linear.py`: fold scores, `C_`, predictions and probabilities;
  the chosen sparsity, coefficients and predictions).

### New estimators: kernel density and factor analysis (#113, batch 6)

- `KernelDensity` in `anofox-ml-neighbors`: exact sums (sklearn's tree
  with `atol = rtol = 0`), the gaussian / tophat / epanechnikov /
  exponential / linear / cosine kernels with sklearn's normalisations,
  sample weights, `Scott` / `Silverman` bandwidths, `score_samples`,
  `score` and `sample`.
- `FactorAnalysis` in `anofox-ml-preprocessing`: sklearn's SVD-based EM
  (exact SVD, `svd_method="lapack"`), varimax / quartimax rotation,
  `transform`, `covariance`, `precision`, `score_samples`. Component signs
  follow the SVD.
- Registry kinds and Python classes; both match scikit-learn 1.8.0
  (`gen_density_fa.py`: 96 KDE configurations to 1e-10 relative, factor
  analysis with the same iterations, likelihoods to 1e-8 and components up
  to sign).

### New estimators: kernel approximation and random projections (#113, batch 5)

- `anofox_ml_preprocessing::kernel_approximation`: `Nystroem` (RBF,
  Laplacian, polynomial, sigmoid and linear kernels), `RbfSampler`
  (random Fourier features), `KernelCenterer`, `GaussianRandomProjection`,
  `SparseRandomProjection` and `johnson_lindenstrauss_min_dim`. Registry
  kinds and Python classes (`RBFSampler` in Python).
- The random estimators draw from their own seeded generator, so their
  features are not sklearn's numbers; tests check what they approximate
  (the kernel, the pairwise distances). `KernelCenterer`,
  `johnson_lindenstrauss_min_dim` and the Nyström feature Gram matrix with
  every row as a component match scikit-learn 1.8.0
  (`gen_kernel_approx.py`).

### New estimators: covariance (#113, batch 4)

- `anofox_ml_preprocessing::covariance`: `EmpiricalCovariance`,
  `ShrunkCovariance`, `LedoitWolf`, `Oas`, the robust `MinCovDet`
  (sklearn's FastMCD: random starts, concentration steps, the subset path
  above 500 rows, the exact univariate case, consistency correction and
  reweighting) and the outlier detector `EllipticEnvelope`. A
  `FittedCovariance` has the location, covariance, precision (scipy's
  `pinvh`), squared Mahalanobis distances and the Gaussian log-likelihood
  `score`. Registry kinds and Python classes (`OAS` in Python).
- Matches scikit-learn 1.8.0 (`gen_covariance.py`): the plain estimators to
  1e-12; MinCovDet and EllipticEnvelope exactly on the support where
  FastMCD reaches the optimal subset. Its random starts come from its own
  generator, and in the univariate case the end of the shortest half that
  enters the support depends on numpy's tie order.

### Changed

- `anofox-ml-linear` and `anofox-ml-preprocessing` use anofox-regression
  0.11 too, so the workspace builds one version of it.

### New estimators: multiclass and threshold meta-estimators (#113, batch 3)

- Registry kinds `OneVsRestClassifier`, `OneVsOneClassifier`,
  `FixedThresholdClassifier` and `TunedThresholdClassifierCV` over any
  registered classifier (`estimator` + `estimator_params`). As in sklearn,
  a binary model's score is its `decision_function` when it has one, else
  `predict_proba[:, 1]`; the threshold classifiers prefer `predict_proba`
  (`response_method`). One-vs-one combines votes with sklearn's confidence
  tie-break; the tuned threshold maximises the mean CV score of
  balanced accuracy / accuracy / F1 / precision / recall over 100
  candidate thresholds.
- Python: `anofox_ml_python.multiclass` (also at the top level).
- `stratified_k_fold_ordered`: sklearn's default, unshuffled
  `StratifiedKFold` (what `cv=k` means for a classifier).
- All four match scikit-learn 1.8.0 over LogisticRegression and GaussianNB
  (`gen_meta.py`, `anofox-ml-registry/tests/golden_meta.rs`).

### New estimators: univariate selection modes and TF-IDF (#113, batch 2)

- `GenericUnivariateSelect` in `anofox-ml-preprocessing` with
  `SelectionMode::{KBest, Percentile, Fpr, Fdr, Fwe}` over the
  `SelectKBest` scoring functions (sklearn's `GenericUnivariateSelect`,
  `SelectPercentile`, `SelectFpr`, `SelectFdr` (Benjamini–Hochberg),
  `SelectFwe`). Python: all five classes.
- `TfidfTransformer` in `anofox-ml-text`: TF-IDF of a count matrix with
  `norm` (L2 / L1 / none), `use_idf`, `smooth_idf` and `sublinear_tf`.
- Both are registry kinds and match scikit-learn 1.8.0
  (`gen_univariate.py`, `tests/golden_univariate.rs`).

### Fixed

- `SelectKBest` breaks ties between equal scores as sklearn does (the
  higher feature indices win), instead of keeping the lower ones.

### New estimators: baselines and simple classifiers (#113, batch 1)

- `DummyClassifier` (`MostFrequent` / `Prior` / `Stratified` / `Uniform` /
  `Constant`) and `DummyRegressor` (`Mean` / `Median` / `Quantile` /
  `Constant`) in `anofox-ml-core`, with sample weights (sklearn's lower
  weighted percentile).
- `NearestCentroid` in `anofox-ml-neighbors`: euclidean / manhattan
  (median) centroids, shrunken centroids (`shrink_threshold`), uniform /
  empirical / given priors, `predict_proba` and `decision_function`.
- `RidgeClassifier` in `anofox-ml-regression`: ±1-encoded ridge, binary
  and multiclass, `class_weight` and sample weights.
- `Perceptron` in `anofox-ml-linear` (SGD with the perceptron loss and a
  constant learning rate, sklearn's defaults).
- All five are registry kinds and Python classes, and match scikit-learn
  1.8.0 (`test_harness/generators/gen_baselines.py`,
  `tests/golden_baselines.rs`).

### Fixed

- `SgdClassifier` / `SgdRegressor` stop on the epoch objective (loss plus
  penalty) as scikit-learn 1.8 does, not on the loss alone. With a
  non-negligible `alpha` the number of epochs, and so the coefficients,
  now match sklearn.

## [0.3.0] - 2026-11-05

Conformal prediction across every surface, PerpetualBooster, quantile loss
for histogram gradient boosting, hyperparameter search (unpublished crate),
and persistence for every fitted model.

### Highlights

- **Conformal prediction** (#93): split, CV+ / jackknife+, normalised, CQR
  and Mondrian intervals; LAC / APS / RAPS and Mondrian prediction sets.
  Typed in `anofox-ml-conformal`, by name in the registry (any registered
  estimator as the base model) and in Python (`anofox_ml_python.conformal`).
- **PerpetualBooster** behind the optional `perpetual` feature.
- **`HistGradientBoostingRegressor(loss="quantile")`** and
  `LgbmObjective::Quantile`.
- **Every fitted type can be saved**: the last seven regression wrappers
  gained serde with anofox-regression 0.11 (#87).
- **Deterministic KMeans** across thread counts (#108).

### Upgrading from 0.2

- `HistGradientBoostingRegressor` has two new public fields, `loss` and
  `quantile`: code that builds it with a struct literal needs
  `..Default::default()` (or the builders). Saved configs still load.
- `LgbmObjective` has a new variant `Quantile { alpha }`: exhaustive
  `match`es need an arm. `LgbmRegressor` now returns an error for the
  classification objectives instead of panicking.
- `DynModel` has two new methods with default implementations
  (`predict_interval`, `predict_set`); `anofox-ml-registry` now depends on
  `anofox-ml-conformal`.
- Python `HistGradientBoostingRegressor` takes `loss` and `quantile` first,
  as sklearn does; pass the other arguments by keyword.
- `KMeans` centroids and inertia, and exact t-SNE, can differ from 0.2 in
  the last digits (they are now summed in a fixed order).
- anofox-regression 0.10 → 0.11 (with its `serde` feature).

### Persistence for the anofox-regression wrappers (#87)

- anofox-regression 0.11 (its new `serde` feature, enabled here). The
  remaining seven wrappers without serde support are now serializable,
  with v2 golden fixtures: `FittedOlsRegressor`, `FittedRidgeRegressor`,
  `FittedLassoRegressor`, `FittedElasticNetRegressor`,
  `FittedWlsRegressor`, `FittedHuberRegressor` and
  `FittedQuantileRegressor`. Inference survives the round trip (an OLS
  `summary()` after loading is identical). Every fitted type in the
  workspace can now be saved.
- The registry keeps its compact form for these kinds, so registry bytes
  and Python pickles saved earlier still load.

### Fixed

- `KMeans` centroids and inertia, and the exact t-SNE gradient, no longer
  depend on the rayon thread count (#108): block partials are summed in
  block order instead of with rayon's `reduce` / `sum`.

### Quantile loss for histogram gradient boosting

- `LgbmObjective::Quantile { alpha }` (LightGBM's `objective="quantile"`):
  pinball-loss gradients, and leaf outputs and the baseline refit to the
  `alpha`-percentile of the residuals (the L1 objective's median refit,
  generalised). The regressor now rejects classification objectives with
  an error instead of panicking.
- `HistGradientBoostingRegressor` gains sklearn's `loss`
  (`SquaredError` / `AbsoluteError` / `Quantile`) and `quantile`
  (`with_loss`, `with_quantile`); Python: `loss="quantile", quantile=q`.
  This closes the gap noted in #93: HGB can be the estimator of a
  `ConformalizedQuantileRegressor` (registry and Python).

### Conformal predictors in the registry (#93, milestone 4)

- `DynModel::predict_interval(x, alpha)` (`n × 2`: lower, upper) and
  `DynModel::predict_set(x, alpha)` (`n × classes`, 1 = in the set). Both
  default to "unsupported", like the other methods. `Out` derives `Debug`.
- `RegistryEstimator { kind, params }` implements `Fit<f64>` for any
  registered estimator. Its `RegistryFitted` implements `Predict` /
  `PredictProba` and serialises as `{kind, bytes}`, so the typed generic
  wrappers can use estimators chosen by name.
- New kinds `SplitConformalRegressor`, `CrossConformalRegressor`,
  `NormalizedConformalRegressor` and `SplitConformalClassifier`. Their
  parameters are `estimator` + `estimator_params` (plus
  `residual_estimator` / `residual_estimator_params` for the normalised
  one) and the wrapper's own settings, e.g. `{"estimator":
  "HistGradientBoostingRegressor", "estimator_params": {"max_iter": 200},
  "calibration_size": 0.25, "seed": 0}`. The default estimators are MAPIE's:
  `OlsRegressor` / `LogisticRegressor`. The results match the typed
  wrappers exactly.
- `ConformalizedQuantileRegressor` (CQR) kind over `QuantileRegressor`
  (default, as in MAPIE), `GradientBoostingRegressor` or
  `PerpetualRegressor`; the registry sets each kind's quantile parameter.
  `alpha` is a fit parameter here; `predict_interval` with another `alpha`
  is an error that asks for a refit.
- `MondrianConformalRegressor` / `MondrianConformalClassifier` kinds
  (group-conditional coverage). The registry `fit` has no group argument,
  so the group id of each row is read from column `group_column` of `x`
  (non-negative integers) at fit and predict time; the estimator sees the
  other columns.
- `SplitConformalClassifier::fit_with_groups`, the classifier counterpart
  of `MondrianConformalRegressor::fit_with_groups`.
- Python: `anofox_ml_python.conformal` with sklearn-style
  `SplitConformalRegressor`, `CrossConformalRegressor`,
  `NormalizedConformalRegressor`, `ConformalizedQuantileRegressor`,
  `MondrianConformalRegressor`, `SplitConformalClassifier` and
  `MondrianConformalClassifier`. They wrap any of the package's
  regressors / classifiers (`estimator=None`: MAPIE's linear defaults);
  `predict_interval(X, alpha)` returns `(n, 2)` bounds, `predict_set(X,
  alpha)` an `(n, n_classes)` boolean array; the Mondrian classes take
  `groups` at fit and predict. `_native.Model` gains `predict_interval` /
  `predict_set`.
- `anofox-ml-registry` now depends on `anofox-ml-conformal`, and
  `scripts/publish.sh` publishes conformal first.

### PerpetualBooster (optional `perpetual` feature)

- `PerpetualRegressor` / `PerpetualClassifier` in `anofox-ml-ensemble`
  wrap the [perpetual](https://crates.io/crates/perpetual) crate
  (Apache-2.0, pinned to `=3.0.0-rc.2`): gradient boosting tuned by a
  single `budget` instead of a learning rate, depth and number of rounds,
  with native missing-value handling. Regression objectives through
  `PerpetualObjective` (squared, absolute, Huber, quantile, Poisson, Gamma,
  Tweedie, ...); binary and multiclass classification with any class labels;
  `sample_weight`. Fitted models persist in perpetual's own JSON format.
- Registered as `PerpetualRegressor` / `PerpetualClassifier` in
  `anofox-ml-registry`, with flat JSON parameters (`{"budget": 1.0,
  "objective": "SquaredLoss"}`).
- Off by default: enable `perpetual` on `anofox-ml`, `anofox-ml-ensemble`
  or `anofox-ml-registry`. It needs Rust 1.88 (perpetual's dependencies),
  above the workspace MSRV of 1.87. Its 2.x releases need nightly Rust,
  which is why the release candidate is used.
- The `registry!` macro accepts attributes such as `#[cfg(...)]` per entry.

### Conformal prediction sets for classifiers (#93)

- `SplitConformalClassifier` / `conformalize_classifier`: prediction sets
  that contain the true class with probability `≥ 1 − α`, with the LAC
  score (smallest sets) or APS (sets grow on hard inputs) in three modes:
  `Include` (conservative), `Exclude` (smaller sets) and `Randomized`
  (coverage `1 − α` on average, seeded). `predict_set` returns an
  `n × classes` membership matrix, `predict_set_labels` the labels;
  `allow_full_sets` instead of an error for too few calibration rows.
- Matches MAPIE 1.6.0's `SplitConformalClassifier` exactly on every set
  (LAC and APS include / exclude, three levels); seeded tests check
  coverage, set size and adaptivity.
- New metrics `set_coverage_score` and `mean_set_size`.
- RAPS (`ClassificationScore::Raps(RapsParams)`, Angelopoulos et al.
  2021): APS with a penalty on classes ranked after `k_reg`, for smaller
  sets when the model spreads mass over many unlikely classes. `k_reg` and
  `lambda` are fixed or tuned per `alpha` on held-out tuning rows
  (`tuning_size`, or `conformalize_raps` with your own split). Matches
  MAPIE 1.6.0 exactly with λ pinned; MAPIE's own λ search has a bug (it
  uses λ in place of k*), which we do not reproduce.
- Mondrian (group-conditional) conformal: `MondrianConformalRegressor`
  (`fit_with_groups`) / `conformalize_mondrian` for split regression, and
  `conformalize_classifier_mondrian` + `predict_set_mondrian` for LAC /
  APS / RAPS sets. The threshold is calibrated within each group (store,
  segment, sensor), so coverage holds per group, not only on average.
- Every fitted conformal model (split, cross, CQR, normalised, Mondrian,
  classifiers) implements `Serialize` / `Deserialize` and round-trips
  through the binary persistence format with identical intervals and sets
  (when its base model is serializable).

### Conformal prediction: CQR and normalised scores (#93)

- `ConformalizedQuantileRegressor` (conformalized quantile regression,
  Romano et al. 2019): quantile models at `α/2`, `1 − α/2` and the median
  are built by a closure (e.g. `GradientBoostingRegressor` with
  `GbLoss::Quantile`), and their band is corrected by the conformal
  quantile of the calibration errors, per bound (default) or symmetric.
  `conformalize_quantiles` calibrates three models you fitted yourself.
- `NormalizedConformalRegressor`: split conformal with residual-normalised
  scores; a second model learns `log |y − ŷ|`, so interval widths follow
  the local error scale. `fit_residual_model` and `conformalize_normalized`
  for prefit models.
- Both match MAPIE 1.6.0 to 1e-9 (5 new golden cases); seeded tests check
  coverage and that the widths track heteroscedastic noise.
- New metric `mean_pinball_loss` (sklearn's).

### Hyperparameter search with rustuna (#92)

- New crate **`anofox-ml-tune`** (not published while
  [optuna/rustuna](https://github.com/optuna/rustuna) is only available from
  git): Optuna-style search over any registry estimator. `tune(kind,
  search_space, x, y, options)` / `tune_with_splits` (custom splitters,
  sample weights) take a JSON search space (`int` / `float` ranges with
  `log` and `step`, `categorical` lists of any JSON values), score each
  trial by cross-validation with a named metric (`r2`, `rmse`, `mae`,
  `accuracy`, `f1_macro`, `roc_auc`, `log_loss`, …; default accuracy for
  classifiers, R² otherwise) and return the best parameters, the trial
  table and the refitted model. Samplers: TPE (default) and random,
  seeded. Failing parameter combinations are recorded as failed trials.
  Multi-objective searches add `objectives` (more metrics, `fit_time`,
  `model_size`) and return the Pareto front; NSGA-II is available as a
  sampler. In a multi-objective search the best trial is the best one on
  the Pareto front. `tune_with_splits_mapped` takes a search space in
  another vocabulary plus a per-trial translation. Pruning waits for
  rustuna pruners.
- Python: `anofox_ml_python.TuneSearchCV(estimator, param_distributions,
  …)` (also in `model_selection`, with the `Int` / `Float` / `Categorical`
  helpers). It searches the estimator's scikit-learn parameters in Rust with
  the GIL released, accepts `cv` as a fold count or any splitter (with
  `groups`), supports `scoring`, `objectives` and the `tpe` / `random` /
  `nsgaii` samplers, and refits a clone of the estimator as
  `best_estimator_`.

## [0.2.0] - 2026-11-04

This release completes the October 2026 algorithm review (#26–#72). It fixes
wrong results and panics, brings every estimator to checked parity with
scikit-learn 1.8 / 1.9, redesigns the core traits around `ndarray` views,
and adopts anofox-regression 0.10. Most of the API surface changed in some
way; the summary below lists what to look at when upgrading, and the
detailed sections follow.

### Highlights

- **Correctness and sklearn parity.** The estimators are checked against
  hardcoded scikit-learn reference values (no parity test is skipped).
  Among the fixes: ROC-AUC / average-precision ties, LDA scalings, PCA,
  SVR / OneClassSVM, DBSCAN / HDBSCAN / LOF / Isomap, random-forest
  feature sampling, LightGBM early stopping, calibration, SGD stopping,
  panics on Fortran-order or NaN input, and HashMap-order nondeterminism.
- **New estimators.** HDBSCAN, BayesianGaussianMixture, MeanShift, OPTICS,
  Birch, SpectralClustering, AffinityPropagation, Isomap, LLE, t-SNE
  (Barnes-Hut), GaussianProcessClassifier, FastICA, CCA, NMF, PLS,
  KernelPCA, TruncatedSVD, RFECV, LocalOutlierFactor, SplineTransformer,
  RcsTransformer, KNNImputer, IterativeImputer, IncrementalPCA and more.
- **Core API.** View-based trait inputs (`fit` / `predict` / `transform`
  accept `&Array2`, `ArrayView2` and slices without a copy); `FitWeighted`,
  `PartialFit`, `PredictProba` and `DecisionFunction` across most
  estimators; `Pipeline` implements `Fit`, is refittable, forwards
  `predict_proba` and accepts supervised steps (`Supervised`); shared
  `ClassWeight`; `sample_weight` and `groups` through the CV helpers.
- **Cross-validation.** Group, stratified-group, purged / combinatorial,
  time-series, spatial-block and buffered splitters, nested CV and
  permutation tests (#74–#79).
- **Regression** delegates to anofox-regression 0.10: multinomial and L1 /
  elastic-net logistic regression, weighted Poisson / Gamma / Tweedie GLMs
  with offsets and opt-in inference, sklearn's Huber and L1 quantile
  regression, LassoLars.
- **Performance.** libsvm's SMO solver with a kernel cache for the kernel
  SVMs, nn-chain Ward clustering, KD-tree neighbours, and faer without its
  thread pool (small fits 5–25× faster).
- **Model persistence.** Binary models are saved as zstd-compressed
  MessagePack with field names (format version 2): smaller files for tree
  models, and saved models survive added, removed and reordered fields.
- **Conformal prediction.** New crate `anofox-ml-conformal`: split
  conformal, CV+ and jackknife+ prediction intervals for any regressor,
  checked against MAPIE.
- **Estimator registry.** New crate `anofox-ml-registry`: every estimator
  by name and JSON parameters, shared by the Python bindings and the
  DuckDB extension.

### Upgrading from 0.1

- **Implementing the traits:** write the `*_view` methods
  (`fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>)`, `predict_view`,
  `transform_view`, ...). Calls to `fit` / `predict` / ... are unchanged.
  Custom pipeline steps (`FitTransform`, `TransformStep`, `FitPredict`,
  `PredictStep`) take views and the pipeline target.
- **Defaults now follow sklearn** (results change without code changes):
  multinomial logistic regression for more than two classes, Huber's joint
  scale, Quantile `alpha = 1.0`, Poisson / Gamma / Tweedie `alpha = 1.0`,
  SVM `gamma = Gamma::Scale` and libsvm tolerances, LDA `Svd` solver,
  NMF `Auto` init, IsolationForest contamination, MiniBatchKMeans batch
  size, tree `max_depth = None`, LightGBM `huber_delta`, GaussianNB
  `var_smoothing`, and more. The sections below mark each one
  **Breaking**.
- **Renamed or reshaped API:** sklearn class names in Python, `Option`
  parameters (`n_components`, `learning_rate`, `bandwidth`, ...), array
  shapes of fitted coefficients (`(K, p)` / `(n, K)`), `ClassicalMDS`,
  removed `seed` fields and the public `kdtree` module.
- **Saving models:** `save_bincode` / `load_bincode` and the other
  `*_bincode*` functions are deprecated aliases of `save_binary` /
  `load_binary` (format version 2); files from 0.1 still load.
- **Random results:** fold assignment, bootstrap draws and tree seeds
  changed, so seeded runs do not reproduce 0.1 numbers.

### Conformal prediction intervals (#93)

- New crate **`anofox-ml-conformal`** (also `anofox_ml::conformal`):
  distribution-free prediction intervals for any regressor.
  `SplitConformalRegressor` (or `conformalize` for a fitted model and your
  own calibration rows) and `CrossConformalRegressor` with the `Plus`
  (CV+, and jackknife+ with leave-one-out splits), `MinMax` and `Base`
  methods. `alpha` is chosen at prediction time; too few calibration rows
  for an `alpha` is an error unless `allow_infinite_bounds` is set.
  Intervals match MAPIE 1.6.0 (absolute conformity score) to 1e-9 in
  golden tests; seeded replications check the empirical coverage.
- New metrics: `coverage_score`, `mean_interval_width` and
  `interval_score` (Winkler score).

### Estimator registry crate

- New crate **`anofox-ml-registry`**: every estimator addressed by a kind
  string and JSON parameters (`make`, `load`, `default_params`, `kinds`,
  the object-safe `DynModel` trait). It was the core of the Python bindings
  and is now shared with the planned DuckDB extension; the Python bindings
  depend on it. A test builds every registered estimator from its default
  parameters and round-trips a fitted model through bytes.
- Fix: `Optics` could not be created from JSON parameters, because its
  infinite `eps` / `eps_max` defaults became `null`. JSON now writes and
  reads `null` as `+∞` for these fields; binary formats are unchanged.

### Model persistence format v2 (#87)

- Binary model files are **format version 2**: the `ANOFOXML` envelope
  header followed by zstd-compressed MessagePack with field names (zstd
  frame checksum included). Saved models now survive changes to the fitted
  types: fields added with `#[serde(default)]` are filled in, removed fields
  are ignored, fields and enum variants may be reordered. Files are smaller
  than before for tree models (a 100-tree forest: 2.28 MB → 0.66 MB);
  loading a forest takes ~9 ms instead of ~4 ms.
- Reading still accepts format version 1 (envelope + bincode, written by
  development versions between 0.1 and 0.2) and bare bincode payloads
  (anofox-ml 0.1); these load while the type's layout is unchanged.
  Older versions cannot read version-2 files. JSON is unchanged.
- New API: `to_binary_bytes`, `from_binary_bytes`, `binary_header`,
  `save_binary`, `load_binary`, `BINARY_MAGIC`, `JSON_FORMAT_VERSION`. The
  `*_bincode*` functions and `BINCODE_MAGIC` are deprecated aliases (they
  write version 2). Python pickles use version 2; older pickles still load.
- `RidgeCrossValidated`, `LassoCrossValidated`, `ElasticNetCrossValidated`,
  `IsotonicRegressor` and `TransformedTargetRegressor` fitted models are
  serializable. `FittedIsotonicRegressor` stores its R² instead of the
  upstream fit. `TransformedTargetRegressor` takes a `TargetTransform`
  (`with_transform`: `Identity`, `Log`, `Log1p`, `Sqrt`); `new(regressor,
  func, inverse_func)` keeps working as `TargetTransform::Custom`, which
  cannot be saved. The remaining OLS / Ridge / Lasso / ElasticNet / WLS /
  Huber / Quantile wrappers wait for serde support in anofox-regression
  (sipemu/anofox-regression#321).
- Golden fixtures (`crates/anofox-ml/tests/fixtures/persistence/v1`, `v2`):
  17 fitted models per format version with their saved predictions; the
  `persistence_compat` test loads all of them.


### faer runs sequentially

- faer is built without its default features (which include `rayon`), as
  in anofox-regression and the anofox-statistics DuckDB extension. faer
  used to spread every decomposition and product over the global rayon
  pool, which made small fits 5–25× slower on a many-core machine (LDA
  5000×20: 16–20 ms → 3 ms, BayesianRidge 19–24 ms → 3 ms; 40× with 20
  idle threads) and erratic under load. Large least squares and PCA are
  unchanged; large dense Cholesky solves (GaussianProcess, KernelRidge with
  thousands of samples) lose up to ~1.4× against faer's best parallel run.
  The estimators still parallelise across trees, folds, classes and
  branches with rayon. Results are unchanged.

### anofox-regression 0.10.0: the #72 blockers are lifted

Every parity test that waited on upstream is enabled; none are ignored now.

- **LassoLars** drops a feature whose coefficient crosses zero, as
  sklearn's `lars_path(method="lasso")`. `active_set` is sklearn's
  `active_` (entry order, re-entries included). Path, knots and `alpha`
  fits match to 1e-10.
- **LogisticRegressor** (#30): multinomial (softmax) loss and L1 /
  elastic-net penalties (`with_l1_ratio`), with sample and class weights,
  through upstream's `MultinomialLogisticRegressor`. Matches sklearn to
  1e-7. **Breaking:** the multiclass default is now
  `MultiClass::Multinomial` (sklearn's default); one-vs-rest is opt-in.
  Wald inference stays available for binary L2 fits. Python
  `LogisticRegression` gains `l1_ratio`.
- **Poisson / Gamma / Tweedie** (#29) implement `FitWeighted` (upstream
  prior weights), plus `fit_weighted_with_offset`. All 18 sklearn
  reference fits (3 families × 3 alphas, weighted and not) match to 1e-6.
  Python passes `sample_weight` through.
- **HuberRegressor**: `HuberScale::Joint` (sklearn's joint coefficient /
  scale estimate; `scale()` / `outliers()` are `scale_` / `outliers_`).
  **Breaking:** it is the new default; `with_scale_method(HuberScale::Mad)`
  gives the previous IRLS at the MAD scale.
- **QuantileRegressor**: L1 penalty `with_alpha` (sklearn's objective,
  exact LP, exact zeros). **Breaking:** `alpha` defaults to sklearn's 1.0
  (Rust and Python); `with_alpha(0.0)` is the previous unpenalised fit.
- **SplineTransformer** (#56, new): sklearn's B-spline expansion through
  upstream's `splines::BSpline` (uniform / quantile / explicit knots, every
  `extrapolation` mode, `include_bias`). Matches sklearn to 1e-12; Python
  `SplineTransformer`.
- BayesianRidge / ARD `predict_std` match sklearn ≥ 1.9, which centres the
  query rows (upstream is correct; sklearn ≤ 1.8 used the raw rows). The
  references are regenerated with sklearn 1.9.1.

### View-based estimator inputs (#37)

- **Breaking:** the core traits take `ndarray` views. Implementors now
  write `fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>)`,
  `predict_view`, `transform_view`, `predict_proba_view`,
  `fit_weighted_view`, `partial_fit_view`, etc. Callers keep using
  `fit` / `predict` / `transform` / ..., which now accept anything that
  views as an array (`&Array2`, `ArrayView2`, slices of a larger
  matrix), so `model.predict(&x)` still compiles and
  `model.predict(x.slice(s![..100, ..]))` no longer needs a copy.
  Weighted entry points keep `Option<&Array1<F>>`; the `*_view` methods
  take `Option<ArrayView1<F>>`.
- Public inherent methods that took `&Array2<F>` as their input
  (`predict_proba`, `decision_function`, `score_samples`, `kneighbors`,
  `predict_std`, ...) accept `impl AsArray` the same way.
- **Breaking:** the pipeline step traits (`FitTransform`,
  `TransformStep`, `FitPredict`, `PredictStep`) take views, and the
  `FunctionTransformer` closure receives an `ArrayView2<F>`. `Pipeline`
  and `FittedPipeline` lose their inherent `fit` / `predict` /
  `predict_proba` / `transform`, which duplicated the `Fit`, `Predict`,
  `PredictProba` and `Transform` impls (import the traits).
- **Breaking:** `as_f64` takes any view and returns a `CowArray`
  (borrowed for `f64`). `LgbmFitOptions::sample_weight` /
  `eval_sample_weight` are `Option<ArrayView1<f64>>`.
- The Python bindings pass borrowed NumPy arrays to the estimators instead
  of copying every input.
- New test `view_inputs`: strided, non-contiguous views give exactly the
  results of their owned copies.

### Correctness parity with sklearn (round 2)

- **anofox-regression 0.9.0** (its API cleanup). The changes here are
  mechanical: `with_intercept` builders, the `BayesianRidgeRegressor` /
  `ArdRegressor` renames, `IterativeFit::iterations`, option setters for
  the now `#[non_exhaustive]` structs, and faer column accessors. No
  results change. The remaining #72 parity tests (LassoLars,
  BayesianRidge / ARD `predict_std`, multinomial / L1 logistic regression)
  still fail with 0.9.0 and stay ignored.

- **anofox-regression 0.8.0.** It fixes several #72 findings:
  - the elastic-net single fit (#187);
  - penalised Poisson / Gamma / Tweedie IRLS convergence;
  - the RANSAC tie-break;
  - Theil-Sen.

  What changes here:
  - Lasso and ElasticNet get an opt-in sklearn penalty scale
    (`with_sklearn_alpha`). It is an exact translation to upstream's raw
    objective, `λ = n·α·(1 + r)` with mixing `2r/(1 + r)` for
    `l1_ratio = r`, and uses a tight tolerance. Python `Lasso` /
    `ElasticNet` use it and now match sklearn (about 1e-7 relative).
  - **Breaking:** `PoissonRegressor`, `GammaRegressor` and
    `TweedieRegressor` default to sklearn's `alpha = 1.0` (Rust and
    Python); `alpha = 0` gives the unpenalised fit. `BinomialRegressor`
    keeps 0.
  - The GLM parity tests at sklearn's alpha are enabled again.
  - Still blocked (#72): sample weights for Poisson / Gamma / Tweedie,
    multinomial / L1 logistic regression, the LassoLars drop rule,
    BayesianRidge / ARD `predict_std`, Huber, the Quantile L1 penalty and
    the B-spline basis.

- **HistGradientBoosting bins like sklearn.** The engine gets
  `BinningMethod::{LightGbm, Sklearn}`, and the HGB facades use sklearn's
  `_BinMapper` thresholds: midpoints between distinct values, or
  percentile midpoints, with a 200 000-row subsample. Before, they used
  LightGBM's greedy bins, so even a single split differed from sklearn.
  - The regressor now matches sklearn to about 1e-7 (sklearn keeps
    gradients in float32).
  - The classifier matches up to exact-gain ties, which sklearn breaks by
    float32 summation order.
  - `LgbmRegressor` / `LgbmClassifier` keep LightGBM's binning.
- **NMF initialisation follows sklearn.**
  - The default is sklearn's `init=None`: `nndsvda`, or `random` when
    `n_components > min(n_samples, n_features)` (which is now allowed with
    a random init). New `NndsvdA` and `NndsvdAr`.
  - The first NNDSVD component uses `|U₀|`, `|V₀|`. Before, a negatively
    signed leading singular vector would have zeroed it.
  - Random init draws `|N(0,1)|·sqrt(mean(X)/k)`, sklearn's distribution.
  - The default fit matches sklearn to 1e-13.
  - Python `NMF` gains `init`, `components_` and `reconstruction_err_`.
  - **Breaking:** the default init changes from `Nndsvd` to `Auto`.
- **FastICA is a port of sklearn's `_fastica`.**
  - Default `Parallel` algorithm with symmetric decorrelation, plus
    `Deflation`.
  - `logcosh(alpha)` / `exp` / `cube` contrasts.
  - `whiten` = unit-variance (default) / arbitrary-variance / none, and
    `svd` / `eigh` whitening solvers.
  - sklearn's `u *= sign(u[0])` eigenvector signs, `w_init` (default
    `N(0,1)`), `mixing_` and `inverse_transform`.
  - Given the same `w_init`, every variant matches sklearn to 1e-8.
  - Before, it ran deflation from a uniform start and scaled sources with
    ddof = 1.
  - **Breaking:** `n_components` is an `Option`, and the fitted fields are
    now `components`, `mixing`, `mean`, `whitening`, `n_iter`.
- **OneClassSVM uses libsvm's dual scaling, as sklearn.** Σα = ν·n.
  `score_samples` is `decision_function + offset()` (sklearn's `offset_`).
  Before, decision values were divided by ν·n and `score_samples` equalled
  `decision_function`. Predictions are unchanged. **Breaking.**
- **KernelPCA eigenvalue cleanup is sklearn's `_check_psd_eigenvalues`.**
  Significantly negative eigenvalues are an error, other negatives become
  0, and positives below 1e-12·max become 0. Rounding noise was kept
  before, so `remove_zero_eig` / Python `n_components=None` returned
  spurious components (79 instead of 4 on rank-4 data). Python
  `n_components=None` now always removes zero eigenvalues, as sklearn.
- **Python `KBinsDiscretizer` defaults to `encode='onehot'`,** as sklearn.
  The output is dense, where sklearn returns a sparse matrix.
- **Isomap follows sklearn.**
  - A disconnected k-NN graph is completed as sklearn's
    `_fix_connected_components` does: an edge between the closest pair of
    every two components. Before, `fit` returned an error.
  - The embedding gets sklearn's KernelPCA eigenvalue cleanup and
    `svd_flip` signs.
  - Embeddings match sklearn to 1e-14 (data with tied k-NN distances can
    differ).
- **LocalOutlierFactor thresholds follow sklearn.**
  - `contamination: Option<f64>`, defaulting to `None` = sklearn's
    `'auto'` (offset -1.5).
  - A numeric contamination uses the linearly interpolated percentile.
  - Outliers are the samples with `nof < offset`.
  - Before: default 0.1, nearest-rank threshold, strict `>`, so about 10%
    of labels differed from sklearn.
  - **Breaking:** the type and default of `contamination` changed.
- **IsolationForest follows sklearn.**
  - Leaf corrections use sklearn's `_average_path_length`, with `c(2) = 1`.
    The general formula gave 0.154 there, which biased all scores.
  - `contamination: Option<f64>`, defaulting to sklearn's `'auto'`
    (offset -0.5). A numeric contamination uses the interpolated
    percentile.
  - Inliers are `score >= offset`. New `decision_function`.
  - Scores agree with sklearn within seed noise.
  - Python gets `max_samples='auto'` / float and `contamination='auto'`.
  - **Breaking:** the default contamination changed.
- **Trees visit features in a random order at every node,** as sklearn's
  splitter does. Splits of equal improvement now go to a random feature
  instead of always the lowest index.
  - The fixed order made forest trees needlessly alike. RandomForest's
    held-out R² on diabetes trailed sklearn by 0.017 (3.6 SE); it now
    agrees within noise.
  - Gradient-boosting rounds each get their own tree seed.
  - Results without ties are unchanged; tie-breaks differ from before.
- **CV folds of LassoCV, ElasticNetCV and RidgeCV(cv=k) are contiguous by
  default,** as sklearn's `cv=int` (`KFold` without shuffling). Before,
  they were shuffled with a seed, so `alpha_` could differ from sklearn's:
  LassoCV predictions on diabetes were 13.8 apart, now 2e-3, which is
  solver tolerance. `with_random_state(seed)` still shuffles. Python
  `ElasticNetCV` defaults to `l1_ratio=0.5`, as sklearn.
- **MiniBatchKMeans is a port of sklearn's algorithm.**
  - k-means++ / random init on `init_size` subsamples, best of `n_init`
    on a validation set.
  - `max_iter` epochs of `batch_size` rows drawn with replacement.
  - Batch-wise weighted-mean centre updates, reassignment of rarely used
    centres (`reassignment_ratio`).
  - Stopping on the exponentially weighted batch inertia
    (`max_no_improvement`), or on `tol · mean(var(X))` when `tol > 0`.
  - `sample_weight`, `labels`, `inertia`, `n_steps`.
  - Before, it applied per-sample online updates with a fixed shift
    tolerance.
  - Mean inertia over seeds agrees with sklearn.
  - **Breaking:** defaults are now sklearn's (`batch_size` 1024, `tol` 0).
- **PassiveAggressiveClassifier follows sklearn's `_plain_sgd`.**
  - Multiclass via one-vs-rest (before: an error for more than 2 classes).
  - Steps use `‖x‖²` without an intercept term.
  - Stopping uses sklearn's rule: 5 epochs whose mean loss exceeds
    `best − tol`.
  - New `decision_function`.
  - With `shuffle=False` it matches sklearn to 1e-12 for PA-I and PA-II.
  - **Breaking:** `coef` is an `Array2`, `intercept` an `Array1`, `classes`
    a `Vec`.
- **Python: classical MDS is exported only as `ClassicalMDS`.** The `MDS`
  alias pointed to classical (Torgerson) MDS, but sklearn's `MDS` is the
  iterative SMACOF algorithm, so the name promised a different method.
  **Breaking:** use `ClassicalMDS`.
- **IsotonicRegressor predicts like sklearn.** Predictions interpolate
  linearly between the fitted points (sklearn's `X_thresholds_` /
  `y_thresholds_`); before, they were a step function. New
  `out_of_bounds` (`Nan` default / `Clip` / `Raise`); before, inputs
  outside the range were always clipped.
- **CCA is a port of sklearn's `CCA`.** It uses NIPALS mode B with
  canonical deflation and `scale`, plus `_svd_flip_1d` signs, and exposes
  weights, loadings, rotations and `n_iter`. Before, it was the
  closed-form SVD solution: the canonical correlations were the same, but
  scores and weights were scaled and signed differently from sklearn's.
  It matches sklearn now. **Breaking:** `FittedCca` fields changed.

### Algorithms — new

- **HDBSCAN** (`anofox-ml-cluster`) with stability-based flat extraction that
  correctly marks low-density outliers as noise.
- **BayesianGaussianMixture** — full variational inference with
  Normal–Wishart conjugate priors (Bishop §10.2), Dirichlet weight prior,
  posterior covariance reporting, ELBO-based convergence.
- **MeanShift, OPTICS, Birch (lite), SpectralClustering, AffinityPropagation**.
- **Isomap, LocallyLinearEmbedding, t-SNE** (exact O(n²) + Barnes-Hut quadtree
  O(n log n) for 2D embeddings).
- **GaussianProcessClassifier** with Laplace approximation, multi-class via
  one-vs-rest. Kernel zoo (Matern, RationalQuadratic, White, Constant, sums,
  products) added on the regressor side.
- **FastICA, CCA, NMF, PLS, KernelPCA + inverse_transform, TruncatedSVD, RFECV**.
- **LocalOutlierFactor** with both brute-force and KD-tree paths
  (auto-dispatch ≤ 20 dims).

### Algorithms — extensions

- **AffinityPropagation** sparse k-NN responsibility matrix path
  (`with_n_neighbors`) for O(n·k) memory and per-iter cost vs the dense
  O(n²) default.
- **t-SNE** Barnes-Hut quadtree path for 2D embeddings, gated by
  `with_method(TSneMethod::BarnesHut)`.
- **KMeans** `sample_weight` via the new `FitUnsupervisedWeighted` trait.
- **GaussianNB** and **SGDClassifier** `partial_fit` for streaming /
  incremental training.
- **GP hyperparameter tuning**: multi-parameter BFGS with finite-difference
  gradients (`optimize_kernel_lbfgs`) — extends the prior single-parameter
  golden-section sweep.

### Core / infrastructure

- New traits: `FitWeighted`, `FitUnsupervisedWeighted`, `PartialFit`,
  `PredictLogProba`, `DecisionFunction`, `RegressorScore`, `ClassifierScore`.
- `CsrMatrix` (compressed sparse row) added to `anofox-ml-core::sparse`. Text
  vectorizers now expose `fit_transform_sparse` returning CSR — typical
  10⁴–10⁵-vocab corpora are dense-infeasible.
- PyO3 bindings extended to cover HDBSCAN, MeanShift, AffinityPropagation,
  BayesianGaussianMixture, LocalOutlierFactor, t-SNE, Isomap,
  LocallyLinearEmbedding.
- GitHub Actions CI: check, test matrix, clippy, fmt, docs, sklearn version
  pin, golden-data sync, Python bindings build, coverage (advisory).
- 35 sklearn-parity validation docs under `validation/sklearn_parity/`, each
  with a Complexity section.

### Fixes

- `RandomForestClassifier` and `ExtraTreesClassifier`: `predict_proba`
  could return `1.0000000000000002` for a unanimous vote, because each
  tree was scaled by `1/n_estimators` before summing; probability metrics
  (`log_loss`) rejected it. The probabilities are now summed and divided
  once, as in scikit-learn.
- `LassoLarsIC` AIC formula corrected.
- Stacking estimators now prefer `PredictProba` when the base estimator
  provides it (matches sklearn behaviour).
- **Cross-validation (#33):** `grid_search_cv`, `randomized_search_cv`,
  `cross_val_predict`, `cross_validate`, `learning_curve` and
  `validation_curve` now work on regression targets. They fall back to seeded
  shuffled K-fold when `y` does not look like class labels, and new
  `*_with_splits` variants accept user-supplied splits. `stratified_k_fold`
  allocates folds like sklearn, so classes smaller than k no longer produce
  empty folds. `stratified_shuffle_split` is deterministic per seed and
  honours `test_size` exactly. `learning_curve` shuffles before taking
  prefixes, and both curves fit once per point.
- **Metrics (#34):** `roc_auc_score` and `average_precision_score` handle
  tied scores like sklearn. `precision_recall_curve` uses sklearn ordering
  (increasing thresholds, decreasing recall, (1, 0) sentinel last); **the
  output order changed**. NaN scores now return an error instead of panicking.
- **RandomForest / ExtraTrees (#45):** `max_features` is sampled per split
  instead of per tree. Defaults are now `Sqrt` for classifiers and all
  features for regressors (sklearn ≥ 1.1). Trees get independent seeds, and
  classifier `predict` averages leaf probabilities. **Breaking:**
  `max_features` is `Option<MaxFeatures>`, and `DecisionTree*` gain `seed`.
- **LGBM (#44):** early stopping bins the eval set with the training bin
  edges and truncates the model to `best_iteration`. `LgbmClassifier`
  supports `eval_set` / `early_stopping_rounds`.
- **CalibratedClassifierCV (#43):** calibrates out-of-fold probabilities or
  decision scores instead of hard labels. Platt scaling uses Newton's method
  (libsvm `sigmoid_train`). It returns the original class labels and rejects
  more than two classes. **Breaking:** the base model must implement
  `PredictProba`, or use `with_decision_function`.
- **HDBSCAN (#57):** now runs the full algorithm (condensed tree, stability,
  EOM / leaf selection) and matches sklearn on golden data. Previously a
  separated third cluster could be labelled as noise. New: `probabilities`,
  `with_allow_single_cluster`, `with_cluster_selection_method`.
- **DBSCAN / OPTICS / LOF / Isomap / AffinityPropagation (#57):**
  - `FittedDbscan::predict` labels new points by the nearest core sample
    (it used to return the training labels).
  - OPTICS and HDBSCAN `min_samples` count the point itself.
  - LOF uses sklearn's `1/(mean + 1e-10)` local reachability density, so
    duplicates score correctly.
  - Isomap errors on a disconnected kNN graph instead of setting the
    geodesic distances to 0.
  - AffinityPropagation exemplars always label themselves; new `converged`
    and `n_iter` fields.
- **Robustness (#42):** clustering, KNN, LOF, trees and ensembles accept
  Fortran-order / transposed input instead of panicking. Majority votes in
  KNN, forests, bagging and voting break ties deterministically toward the
  smallest label. Sorting and binning no longer panic on NaN.
  RobustScaler, QuantileTransformer and KBinsDiscretizer ignore NaN at fit
  and pass it through at transform.
- **PCA (#51):** now uses faer (thin SVD, or covariance eigendecomposition
  for tall data) instead of power iteration, and matches sklearn components
  and signs. Adds `explained_variance_ratio`, `singular_values`,
  `noise_variance` and `whiten`. TruncatedSVD, FastICA, NMF and CCA use thin
  SVD; TruncatedSVD signs and explained variance match sklearn.
- **Text (#52):**
  - `CountVectorizer::fit` / `TfidfVectorizer::fit` return fitted
    vectorizers whose `transform` encodes new documents.
  - HashingVectorizer uses MurmurHash3 like sklearn, so `alternate_sign`
    actually cancels collisions. **Its output changes.**
  - The tokenizer is Unicode-aware (`(?u)\b\w\w+\b`).
  - `min_df > max_df` now returns an error.
- **LDA (#65):** `scalings` / `transform` were computed from the wrong
  matrix. They now match sklearn `solver='eigen'`; new
  `explained_variance_ratio`.
- **SVM (#63):**
  - SVR and OneClassSVM project exactly onto the dual feasible set.
  - LinearSVC uses liblinear dual coordinate descent with a `loss` option
    (default squared hinge, as sklearn).
  - LinearSVR uses liblinear dual coordinate descent with L1 / L2
    ε-insensitive loss.
  - LinearSVC and LinearSVR coefficients match sklearn.
- **SGD (#31):**
  - sklearn stopping rule (`tol`, `n_iter_no_change`, `early_stopping` /
    `validation_fraction`) and `n_iter()`.
  - `partial_fit` runs one epoch per call and keeps the learning-rate
    schedule; `SgdRegressor` implements `PartialFit`.
  - Stable log loss, L1 gives exact zeros, and new `average` (ASGD) and
    `predict_proba`.
  - **Breaking:** binary classifiers train a single model
    (`weights().len() == 1`).

### sklearn parity (review P1)

Each item below is checked against hardcoded scikit-learn 1.8 reference values.
**Breaking** marks API or default changes.

- **anofox-regression 0.7 (#27):**
  - The dependency is bumped from 0.5 to 0.7.
  - TheilSen, RANSAC, BayesianRidge/ARD (now with `predict_std`), LARS /
    LassoLars / LassoLarsIC, PassiveAggressiveRegressor (now with `PartialFit`)
    and PLSRegression (now with `transform` and scores) delegate to it.
  - LassoLarsIC evaluates sklearn's criterion along a single path.
  - RANSAC no longer panics when `min_samples > n` or on NaN.
  - Upstream deviations from sklearn are tracked in #72 (no local
    workarounds): the LassoLars drop rule and `predict_std` centring.
- **Regularisation CV (#28):** LassoCV/ElasticNetCV use warm-started
  coordinate-descent paths on sklearn's alpha scale with the default 100-alpha
  grid. They select by pooled MSE over seeded or explicit folds and expose
  `alphas()`, `mse_path()` and `alpha_1se()`. RidgeCV defaults to exact
  leave-one-out from one SVD. **Breaking:** the alpha scale and the folds changed.
- **GLMs (#29):**
  - Poisson/Tweedie/Gamma/Binomial `alpha` now has sklearn's scaling
    (upstream `lambda = alpha · Σw`).
  - Offset/exposure support and opt-in inference via `with_inference(true)`
    → `summary()` (SE, z/t, p, CI).
  - `predict_interval` for OLS/WLS/Ridge.
  - Binomial `sample_weight` and probit/cloglog links.
  - Blocked on anofox-regression (#72): sklearn's default `alpha = 1.0`
    (upstream penalised IRLS does not converge; the default stays 0) and
    `sample_weight` for Poisson/Tweedie/Gamma.
- **LogisticRegressor (#30):**
  - Binary and one-vs-rest multiclass with `C`, `class_weight` and
    `sample_weight`, delegated to upstream's logit GLM.
  - Implements the core classifier traits.
  - **Breaking:** `predict_proba` is `(n, K)` and the default is `C = 1.0`.
  - Blocked on anofox-regression (#72): the multinomial loss and L1 /
    elastic-net penalties (both return an error).
- **Metrics (#35):**
  - `Average::Binary { pos_label }`, per-class scores, and `ClassificationOptions`
    (`labels`, `sample_weight`, `zero_division`).
  - Multiclass `roc_auc_score_multiclass` (OvR/OvO) and `log_loss_multiclass`.
  - Labels are encoded once with exact comparison.
  - With constant `y`, `r2_score` gives 1.0 for a perfect fit and 0.0 otherwise.
- **Halving search (#36):** per-round (stratified) k-fold CV, sklearn's schedule,
  and NaN scores ranked last. Also returns per-iteration `n_resources` /
  `n_candidates`.
- **Boosting / stacking (#46):**
  - GradientBoostingClassifier uses Newton leaf values and true multinomial deviance.
  - AdaBoost SAMME uses sample weights; AdaBoost.R2 predicts the weighted median.
  - StackingClassifier uses stratified shuffled folds and drops the binary column.
- **HGB / LGBM (#47):**
  - HGB handles NaN natively, with a learned split direction.
  - `max_bins` must be ≤ 255 (**breaking:** HGB default is now 255).
  - LGBM monotone constraints now hold globally.
- **Feature selection (#53):**
  - RFECV scores one elimination path per fold with a train/test scorer.
  - SelectFromModel uses absolute importances and a mean/median threshold.
  - k-NN mutual information (`mutual_info_regression` / `mutual_info_classif`).
  - SelectKBest gains chi2, MI scorers and p-values.
  - **Breaking:** the RFECV scorer signature changed and
    `MutualInformationSelector` lost `n_bins`.
- **Transformers (#54):**
  - OneHotEncoder: real categories, `handle_unknown`, `drop`, `min_frequency` /
    `max_categories`, `inverse_transform` and sparse output.
  - OrdinalEncoder: `UseEncodedValue`.
  - QuantileTransformer: two-sided interpolation and inverse.
  - PowerTransformer: Brent-optimised λ and Box-Cox.
  - KBinsDiscretizer: kmeans strategy and duplicate-edge removal.
  - VarianceThreshold / StandardScaler: sklearn constant-feature handling.
- **KMeans (#58):** `n_init`, `KMeansInit`, data-relative `tol`, and empty-cluster
  relocation. One shared greedy k-means++ is used by MiniBatchKMeans, GMM and
  BGMM. Assignment uses GEMM plus rayon.
- **Gaussian mixtures (#59):**
  - Tied/Spherical covariance types and k-means initialisation.
  - The precision Cholesky is computed once per iteration.
  - New `score_samples` / `bic` / `aic` / `converged`.
  - BGMM `lower_bound` is now the real ELBO.
  - A singular covariance is now an error.
- **SVM (#64):**
  - Multiclass SVC/NuSVC use one-vs-one, with `decision_function_shape` and
    `break_ties`.
  - `class_weight` and `probability = true` (Platt + pairwise coupling).
  - Sigmoid and precomputed kernels.
  - **Breaking:** `gamma` defaults to `Gamma::Scale`, and the polynomial kernel
    takes `gamma`.
- **Naive Bayes (#67):**
  - **Breaking:** GaussianNB `var_smoothing` is relative to the largest feature
    variance.
  - GaussianNB gains `priors`.
  - Multinomial/Bernoulli gain `partial_fit`, `sample_weight` and
    `fit_prior`/`class_prior`.
  - `predict_log_proba` is computed in log space.
  - New ComplementNB and CategoricalNB.
- **Gaussian processes (#68):**
  - **Breaking:** `fit` optimises kernel hyperparameters (analytic LML gradients,
    bounded L-BFGS-B in `anofox_ml_core::optim`).
  - ARD kernels and per-parameter bounds.
  - New `predict_cov` / `sample_y`.
  - The White kernel matches sklearn.
  - The GPC objective bug is fixed, and GPC uses erf-based probabilities.
- **MLP (#69):**
  - **Breaking:** sklearn L2 scaling, stopping rule, Glorot initialisation and a
    single logistic output for binary problems.
  - New lbfgs solver.
  - SGD momentum/Nesterov with learning-rate schedules.
  - Early stopping, `partial_fit` / warm start and multi-output regression.

### Cross-validation designs (#74–#79)

New `anofox_ml_core::splitters` and `anofox_ml_core::validation` modules.
Every splitter returns `Vec<CvSplit>` for the `*_with_splits` helpers.

- **Group splitters (#74):**
  - `group_k_fold` now balances folds like sklearn (largest group to the
    lightest fold). **Breaking:** fold assignment changed.
  - New `group_k_fold_shuffled`, `stratified_group_k_fold`,
    `leave_one_group_out`, `leave_p_groups_out` and `group_shuffle_split`.
  - Unshuffled splits match sklearn 1.8.
- **Purged CV (#75):** `purged_k_fold` and `combinatorial_purged_cv` (López de
  Prado) for labels spanning a time interval, with purging and embargo. CPCV
  also returns the backtest paths.
- **Spatial CV (#76):** `spatial_block_k_fold` (grid blocks), `spatial_buffer`
  for any splits, and `buffered_leave_one_out`, with Euclidean or haversine
  distance.
- **`permutation_test_score` (#77):** p-value of a CV score against label
  permutations, optionally within groups.
- **Bootstrap validation (#78):** `bootstrap_splits` (+ stratified) and
  `bootstrap_validate`, which returns the apparent, out-of-bag, .632, .632+
  and optimism-corrected errors.
- **From the sister crates (#79):**
  - `time_series_split_with(TimeSeriesSplitOptions { test_size, gap,
    max_train_size, min_train_size })`, matching sklearn.
  - `group_time_series_split` and `quantile_strata` /
    `stratified_k_fold_continuous` for continuous targets.
  - `nested_cross_validate`.

### Review P2 / P3

- **Kernel SVMs (#62, #64):** SVC, NuSVC, SVR, NuSVR and OneClassSVM share a
  port of libsvm's SMO solver (second-order working-set selection, gradient
  maintenance, shrinking, and an LRU kernel column cache bounded by
  `cache_size` MB). The full Gram matrix is no longer built.
  - NuSVC solves the ν dual directly instead of bisecting C.
  - SVC/NuSVC decision values match sklearn exactly.
  - A degenerate dual (e.g. ν too large) returns an error, as sklearn does.
  - **Breaking:** defaults follow libsvm: `tol = 1e-3`, `max_iter` is an SMO
    iteration cap (default 10⁷), and there are new `shrinking` / `cache_size`
    fields.

- **LDA / QDA (#66):**
  - LDA: sklearn `solver` `Svd` (default; handles collinear features),
    `Lsqr` and `Eigen`; `Shrinkage::{Fixed, Auto}` (Ledoit-Wolf on
    standardised data); `priors`; sklearn `transform` semantics.
  - QDA: per-class SVD with sklearn's `reg_param` (`(1 − r) S² + r`),
    `priors`, and batched scoring (no per-sample factorisation).
  - New `ledoit_wolf` and `oas` estimators. All match sklearn 1.8 to 1e-9.
  - **Breaking:** the default solver is now `Svd`; LDA's `shrinkage` is a
    `Shrinkage` enum; QDA's fitted model stores `rotations` / `scalings`
    instead of `sigmas`, and `with_reg` means sklearn's `reg_param`
    (default 0).

- **NMF / KernelPCA (#55):**
  - NMF has sklearn's coordinate-descent solver (default) and
    multiplicative updates for the Frobenius, Kullback-Leibler and
    Itakura-Saito losses, plus `alpha_W` / `alpha_H` / `l1_ratio`
    regularisation.
  - The NMF convergence check no longer forms WH.
  - `w` (the `fit_transform` result) is stored, and `transform` is
    deterministic and implements `Transform`.
  - NNDSVD zeroes entries below 1e-6, as sklearn does.
  - The CD and KL-MU results match sklearn 1.8.
  - KernelPCA builds the kernel with one GEMM, clips non-positive
    eigenvalues, projects zero-eigenvalue components to 0, adds
    `remove_zero_eig`, uses sklearn's sign convention, and matches sklearn
    to 1e-9.

- **Consistency (#41):**
  - New `anofox_ml_core::labels` (`encode_labels`, exact total-order label
    comparison, shared by metrics).
  - CSV reader: empty and `NA` fields are read as NaN, returned headers
    match the columns of X, and `CsvError` converts into `RustMlError`.
  - `CsrMatrix::from_triplets` checks bounds; new `try_from_triplets`.

- **Parallelism (#38):**
  - New `grid_search_cv_with_splits_par` and `cross_val_score_with_splits_par`
    evaluate in parallel with rayon and give results identical to the
    sequential versions.
  - Multi-output targets and ColumnTransformer branches are fitted in
    parallel.
  - `permutation_importance` copies X once per feature instead of once per
    repeat.
  - Row and column selection use ndarray's `select`.

- **Regression conversions and small solvers (#32):**
  - ndarray → faer conversion copies standard-layout arrays through a
    strided view instead of bounds-checked indexing; OLS/WLS, Ridge,
    ElasticNet and Quantile predict in ndarray from cached coefficients
    (aliased / `NaN` coefficients count as 0, as upstream).
  - Weighted Ridge delegates to the upstream solver (weighted centring and
    `√w` row scaling, as sklearn does) and matches sklearn 1.8.
    **Breaking:** `FitWeighted` returns `FittedRidgeRegressor`;
    `FittedWeightedRidgeRegressor` is removed.
  - OMP grows the Cholesky factor of the active Gram matrix by one row per
    step (sklearn's `_cholesky_omp`), stops on linear dependence and
    matches sklearn 1.8. **Breaking:** the default is `max(⌊0.1·d⌋, 1)`
    coefficients, `tol` overrides `n_nonzero_coefs`, and
    `n_nonzero_coefs > d` is an error.
  - New `SvmKernel::matrix` / `SvmKernel::gram` build kernel matrices from
    one GEMM; KernelRidge and SVC prediction use them.

- **Weights, class weights and pipelines (#37):**
  - New shared `anofox_ml_core::ClassWeight` (`Balanced`, `Manual`) with
    `compute_sample_weight`, `effective_sample_weight` and
    `check_sample_weight`. Trees, SVC, logistic regression, LGBM and SGD use
    it. **Breaking:** `svm::ClassWeight::Map` and
    `regression::ClassWeight::Custom` are now `ClassWeight::Manual`;
    `LgbmClassWeight` is an alias of the shared type.
  - `FitWeighted` for decision trees, random forests, extra trees, gradient
    boosting, AdaBoost, LGBM and SGD. Weighted gradient boosting, AdaBoost
    and SGD match sklearn 1.8 exactly.
  - Random-forest `class_weight` is computed on the full target, not on each
    bootstrap draw (sklearn's `balanced`). **Breaking:** forest results with
    `class_weight` change.
  - Weighted tree splits use running weighted sums (O(n) per feature instead
    of O(n²)); unit weights reproduce the unweighted tree exactly.
  - SGD's `optimal` and `invscaling` learning-rate schedules now follow
    sklearn (`t₀ = 1/(typw·α)`, `eta0 / t^power_t`). **Breaking:** results
    with these schedules change. `SgdClassifier` gains `class_weight`.
  - `Pipeline::fit` borrows the pipeline (refit per CV fold), and `Pipeline`
    implements `Fit`; `FittedPipeline` implements `Predict`, `Transform` and
    `PredictProba` (with the new `Pipeline::set_classifier`).
  - The target reaches every pipeline step: `FitTransform::fit_transform`
    takes `y: Option<&Array1<F>>`, and the new `Supervised(...)` adapter puts
    `Fit`-based transformers (`SelectKBest`, `Rfe`, `Rfecv`,
    `SequentialFeatureSelector`, which now implement `Fit` / `Transform`)
    into a `Pipeline`, `FeatureUnion` or `ColumnTransformer`.
    `FitUnsupervised` gains the provided hook `fit_with_target`.
    **Breaking:** custom `FitTransform` impls take the extra argument.
  - `ColumnTransformer` and `FeatureUnion` compute each branch's training
    output once instead of fitting and transforming again.
  - Unneeded `unsafe impl Send/Sync` removed from the pipeline types.
  - New `cross_validate_weighted_with_splits` /
    `cross_val_score_weighted_with_splits` pass the fold's weights to the fit
    and to the scorer.
  - Not done: view-based (`ArrayView2`) trait inputs.

- **Persistence (#39):**
  - Saved models carry a versioned envelope (`format`, `format_version`,
    `crate_version`, `type_name`, `payload`; bincode files start with the
    magic bytes `ANOFOXML`). Loading checks it, so a file read as the wrong
    type or written in a newer format fails with a clear error. Type names
    are compared without module paths. Files without an envelope still
    load.
  - New `ModelHeader`, `json_header` / `bincode_header` (read the header
    only), `short_type_name` and `ModelBundle { config, fitted }`.
  - Python `save_json` / `save_bincode` store the estimator's configuration
    with the fitted model, so `load_json` / `load_bincode` restore the
    hyperparameters instead of the defaults (older files still load, with
    default hyperparameters).
  - Pipelines with boxed steps remain non-persistable (documented).

- **Python bindings (#40):**
  - Rebuilt on one generic native class (`_native.Model`) backed by a Rust
    registry of 95 estimators: linear and GLM regression, logistic,
    SGD, passive-aggressive, trees, forests, extra trees, gradient boosting,
    AdaBoost, bagging, HistGradientBoosting, LGBM, isolation forest, SVMs,
    naive Bayes, k-NN, LOF, LDA/QDA, MLP, the scalers / imputers /
    decompositions / selectors, clustering and manifold learners.
  - A Python layer gives them sklearn's API: `get_params` / `set_params`,
    `clone` / `GridSearchCV` / pipeline compatibility (`__sklearn_tags__`),
    `fit` returning `self`, `random_state`, `predict_proba` /
    `decision_function` / `score_samples` where available, `sample_weight`,
    string labels, `NotFittedError`, pickling (joblib / multiprocessing)
    through the versioned bincode envelope.
  - Fitting and prediction release the GIL.
  - More metrics, `train_test_split`, `KFold`, `cross_val_score`.
  - CI builds the wheel and runs the Python test suite.
  - **Breaking:** sklearn class names (`KNeighborsClassifier`, `SVC`, `PCA`,
    `TSNE`, ...; the old names remain as aliases), `random_state` instead of
    `seed`, and `fit` returns `self`. The old `save_json` / `load_json`
    methods are replaced by pickling.
  - `Lasso` / `ElasticNet` use anofox-regression's penalty scaling, which
    does not yet match sklearn's `alpha` (#72).
- **Tree and boosting performance (#48):**
  - Decision trees presort every feature once per tree and split nodes by
    stable partition (no per-node sorting); classes are encoded once per
    tree instead of hashed per sample move. Trees are byte-identical to
    before and fit 2–2.7× faster.
  - New `DecisionTreeClassifier::fit_on_rows` / `DecisionTreeRegressor::
    fit_on_rows` fit on (bootstrap) row lists without copying `X`; random
    forests, bagging, gradient-boosting subsampling and AdaBoost.R2 use
    them (same results, no per-tree copy of `X`).
  - LGBM: histogram subtraction (only the smaller child's histograms are
    built), per-feature bin counts, column-major bins for histograms and
    row-major bins for prediction, no per-row allocations, O(1) categorical
    routing, histograms built in parallel for large nodes. Fitting is
    3.5–4× faster with the same predictions.
  - Weighted splits use running weighted sums (see #37) and the RF OOB score
    uses an in-bag mask.

- **HistGradientBoosting on the LGBM engine (#49):**
  - `HistGradientBoostingRegressor` / `HistGradientBoostingClassifier` are
    facades over the LGBM engine with sklearn's defaults
    (`max_leaf_nodes = 31`, `max_depth = None`, leaf-wise growth). They gain
    native missing-value handling, `FitWeighted`, feature importances and
    the engine's speed-ups; the separate histogram implementation is gone.
  - **Breaking:** `max_depth` is `Option<usize>` (default `None`), there is
    a new `max_leaf_nodes`, results change, and fitted HGB models saved by
    earlier versions do not load.
  - Ensemble helpers are shared instead of duplicated (`majority_vote`,
    `select_rows`, the boxed-estimator traits of voting / stacking); the
    tree classifier and regressor share one tree engine (#48).

- **Tree / ensemble parity, part 1 (#50):**
  - Decision trees gain `min_impurity_decrease`, `min_weight_fraction_leaf`,
    `max_leaf_nodes` (best-first growth) and `ccp_alpha` (minimal
    cost-complexity pruning); all match sklearn 1.8.
  - Leaves store their impurity, weighted sample totals and exact
    per-class weights (previously weighted counts were rounded to
    `w·1000`); feature importances use weighted node sizes and leaf
    impurities, as sklearn. Older saved trees still load.
  - ExtraTrees: `criterion` (Gini / Entropy) and an `Array2`
    `predict_proba` (plus the `PredictProba` trait). **Breaking:** the
    inherent `predict_proba` returned `Vec<Vec<(F, F)>>`.
  - GradientBoostingRegressor: `AbsoluteError`, `Huber` and `Quantile`
    losses with sklearn's terminal-region updates, early stopping
    (`n_iter_no_change`, `validation_fraction`, `tol`) and `max_features`.
  - GradientBoostingClassifier gets the same early stopping (on a
    stratified hold-out, scored by the binomial / multinomial log loss) and
    `max_features`. In both estimators each round's tree now draws its
    features with its own seed. Before, every round reused one seed, so all
    trees drew the same feature sequence.
  - Bagging{Classifier,Regressor}: `max_features` / `bootstrap_features`
    (one feature subset per estimator, as sklearn), `oob_score` with
    `oob_decision_function` / `oob_prediction`, `FitWeighted`, and
    per-estimator tree seeds. **Breaking:** `BaggingClassifier::predict` is
    the argmax of the averaged probabilities (sklearn) instead of a majority
    vote.

- **LGBM parity, part 2 (#50):** checked against LightGBM 4.7.
  - The L1 objective starts from the (weighted) median and refits every
    leaf to the median residual of its rows (LightGBM's
    `RenewTreeOutput`), with `sign(0) = 0` gradients. Previously it used the
    mean and plain Newton leaves.
  - Baselines use sample weights: the weighted mean (L2 / Huber), the
    weighted positive rate (binary) and weighted class priors (multiclass).
  - LgbmClassifier's softmax hessian carries LightGBM's `K / (K - 1)`
    factor. The HistGradientBoosting facade keeps sklearn's hessian, and
    its multiclass output matches sklearn.
  - Categorical binning keeps the most frequent categories, at most 254 and
    none rarer than `min_data_in_bin`. Every other category (and any
    category unseen in training) goes to the "other"/missing bin. Before,
    categories beyond 254 collapsed into the last bin, and unseen
    categories landed on a real category.
  - Categorical splits follow LightGBM: one-vs-rest up to
    `max_cat_to_onehot`, otherwise `G / (H + cat_smooth)` ordering scanned
    from both ends with `cat_l2`, `max_cat_threshold` and
    `min_data_per_group` (new `CategoricalSplitParams`, LightGBM defaults).
    Missing and "other" categories always go right.
  - **Breaking:** `huber_delta` defaults to LightGBM's 0.9 (was 1.0), and
    results change for the L1 objective, weighted fits, multiclass LGBM and
    categorical features.

- **Neighbours, part 1 (#60):**
  - New `NeighborIndex` (sklearn's `NearestNeighbors`): batched, parallel
    `kneighbors`, `kneighbors_self` (each sample excluded from its own
    list), `radius_neighbors` and `radius_neighbors_self`.
    `NeighborAlgorithm::Auto` follows sklearn's rule (brute force for
    non-Minkowski metrics, more than 15 features or
    `k >= n_samples / 2`). Results are exact and identical across
    algorithms, with ties going to the lower index.
  - The KD-tree is rewritten. It stores a flat permuted copy of the data,
    uses leaves of 30 points and splits at the median of the widest
    dimension (`select_nth_unstable`). It prunes with node bounding boxes,
    so it serves every Minkowski metric. Building it no longer sorts at
    every level.
  - Euclidean brute force pre-selects candidates with a matrix product,
    then ranks them by exact distances (about 4× faster, still exact).
  - New metrics `Chebyshev`, `Minkowski(p)` and `Precomputed`.
  - kNN estimators: `algorithm`, `kneighbors`, `predict_proba` /
    `PredictProba` for the classifier. `weights = Distance` uses sklearn's
    rule: neighbours at distance 0 take all the weight, replacing the old
    `1e15` cap.
  - New `RadiusNeighborsClassifier` (with `outlier_label`) and
    `RadiusNeighborsRegressor` (NaN without neighbours), also in the
    Python bindings, which gain `p`, `algorithm` and the new metrics.
  - LOF uses the shared index.
  - All of this matches sklearn 1.8 (Rust and Python parity tests).
  - **Breaking:** the public `kdtree` module is gone, and fitted kNN
    models saved by earlier versions do not load.

- **Neighbours, part 2 (#60):** clustering and manifold learning query the
  shared `NeighborIndex` instead of building dense n×n distance matrices.
  - DBSCAN computes every eps-neighbourhood up front and expands clusters
    depth-first like sklearn's `dbscan_inner`, so border points get
    sklearn's labels. It gains `metric` (including `Precomputed`) and
    `algorithm`; `predict` queries the index of core samples.
  - OPTICS takes core distances from the index and picks the next point
    from a heap with lazy deletion. Before, a linear scan plus
    `stack.contains` made it about O(n³). It now exposes `core_distances`
    and `predecessor`; ordering, reachability and labels match sklearn.
  - HDBSCAN takes core distances from the index, and its Prim MST
    computes distances on the fly (O(n) memory instead of n²).
  - SpectralClustering (k-NN affinity), Isomap and LLE build their k-NN
    graphs with the index. Isomap's geodesics use parallel Dijkstra on the
    sparse graph (sklearn's `path_method="D"`) instead of O(n³)
    Floyd–Warshall, and its embedding matches sklearn.

- **t-SNE (#61)** follows sklearn's `TSNE`:
  - The optimiser uses per-coordinate gains (+0.2 / ×0.8, minimum 0.01).
    It runs 250 exploration iterations with momentum 0.5 and exaggerated P,
    then momentum 0.8. KL is checked every 50 iterations, with
    `n_iter_without_progress` and `min_grad_norm` stopping. Before, there
    were no gains, exaggeration lasted `n_iter / 4` iterations, and the
    momentum switch was fixed at 250.
  - `learning_rate: Option<f64>` (`None` = `"auto"`), `max_iter`
    (default 1000; `n_iter` still deserialises), and `init`: `TSneInit::Pca`
    by default, or `Random` / `Array`.
  - Perplexities are searched over single-precision squared distances, as
    sklearn does.
  - Barnes–Hut is the default. It uses a sparse P from the
    `⌊3·perplexity + 1⌋` nearest neighbours (no more dense n×n P or
    distance matrix) and a port of sklearn's space-partitioning tree, so
    it works for 1–3 output dimensions instead of silently falling back to
    exact. It runs in parallel.
  - The affinities, the first objective and the first optimiser steps
    match sklearn to rounding; full runs match its iteration counts and
    mean KL. On 3,000 points Barnes–Hut takes 2.8 s (sklearn: 4.2 s) and
    reaches the same KL.
  - **Breaking:** `learning_rate` is an `Option`, `n_iter` is `max_iter`,
    and the defaults change (Barnes–Hut, PCA init, automatic learning
    rate, 1000 iterations). Perplexity must be below `n_samples`; it was
    silently clamped before.

- **LLE and spectral clustering (#61):**
  - New `sparse_eigen::smallest_eigenpairs` (manifold crate). It uses a
    dense eigendecomposition up to n = 500, and above that shift-invert
    Lanczos (faer's thick-restart Lanczos on `(M + εI)⁻¹` through a sparse
    Cholesky factorisation, like sklearn's ARPACK with `sigma`).
  - LLE regularises with sklearn's `reg · trace(G)`; it was
    `reg · trace / k`, k times weaker. It builds the sparse
    `(I − W)ᵀ(I − W)` in O(n·k²) instead of a dense O(n³) product, uses the
    new eigensolver, and reports `reconstruction_error`. Embeddings match
    sklearn for n = 100 and n = 800.
  - New `spectral_embedding` (sklearn's `manifold.spectral_embedding`).
  - SpectralClustering now follows sklearn. The k-NN affinity is
    `0.5·(C + Cᵀ)` with self loops, replacing an OR of the two directions.
    The embedding is divided by √degree, with sklearn's deterministic sign
    flip, instead of being row-normalised. k-means gets `n_init` restarts
    (default 10). Large k-NN graphs use the sparse eigensolver. Python
    gains `affinity="nearest_neighbors"`, `n_neighbors` and `n_init`.

- **AgglomerativeClustering (#61)** builds the tree like sklearn.
  - Single linkage uses Prim's MST with O(n) memory.
  - Complete, average and Ward use scipy's nearest-neighbour chain with
    Lance–Williams updates, O(n²) in total. Before, single, complete and
    average ran a naive O(n³) loop.
  - Merges are sorted and relabelled, exposing `children` and `distances`.
  - Labels come from a port of sklearn's `_hc_cut`, so the label numbering
    matches too.
  - New `distance_threshold` (also in Python).
  - The `RUSTML_AGGLO_NAIVE` environment switch is removed.
  - Children, distances and labels match sklearn exactly for all four
    linkages.

- **ClassicalMds (#61)** follows sklearn 1.8's `ClassicalMDS`. It gains a
  `metric`, including `Precomputed` dissimilarity matrices, its main use.
  It uses sklearn's sign convention. With Euclidean features it solves the
  `d × d` covariance eigenproblem (O(n·d²)) instead of an n×n one (O(n³)).
  It matches sklearn for Euclidean, Manhattan and precomputed input; Python
  adds `metric` and a `ClassicalMDS` alias.

- **MeanShift (#61)** follows sklearn.
  - `bandwidth: Option<f64>` (`None` estimates it with the new
    `estimate_bandwidth`).
  - New `seeds`, `bin_seeding` / `min_bin_freq` and `cluster_all`.
  - The stopping threshold is `1e-3 · bandwidth`.
  - Seeds shift in parallel through the neighbour index, instead of
    copying every row on every shift.
  - Centres are sorted by intensity and suppressed within the bandwidth,
    where before the first-found centre won within bandwidth / 2.
  - Centres, labels and `n_iter` match sklearn exactly.
  - **Breaking:** `bandwidth` is an `Option` and `tol` is gone.

- **OPTICS ξ extraction (#61):** new `OpticsClusterMethod::Xi`
  (`Optics::xi`), a port of sklearn's `cluster_optics_xi`. It covers steep
  areas, `min_cluster_size` and predecessor correction, and exposes
  `cluster_hierarchy`. Labels and hierarchy match sklearn exactly. DBSCAN
  extraction stays the Rust default; the Python `OPTICS` now defaults to
  ξ, like sklearn.

- **Birch (#61)** is now a real CF-tree, ported from sklearn.
  - A sample descends to the closest subcluster and merges when the merged
    radius stays within `threshold` (before: distance to the centroid).
  - Nodes split around their farthest pair, up to new roots
    (`branching_factor`, default 50).
  - Subcluster centres come out in leaf order.
  - The global step is AgglomerativeClustering, as in sklearn; it was
    KMeans, which ignored subcluster sizes.
  - Centres, subcluster labels and sample labels match sklearn exactly in
    the parity tests. On other data the leaf order can differ on ULP-level
    ties, but the partition is the same.
  - **Breaking:** the `seed` field is gone.

- **Incremental preprocessing (#56):** new core trait
  `PartialFitUnsupervised` (`partial_fit(state, X)`, plus
  `partial_fit_weighted`).
  - `StandardScaler` merges batches with sklearn's
    `_incremental_mean_and_var`, per feature and ignoring NaN. It takes
    `sample_weight` through `FitUnsupervisedWeighted` and
    `partial_fit_weighted`, and exposes `var()` and `n_samples_seen()`.
  - `MinMaxScaler` and `MaxAbsScaler` implement `partial_fit`.
  - New `IncrementalPca`, a port of sklearn's `IncrementalPCA`: batch SVD
    with mean correction; `fit` runs in batches of `batch_size` rows.
  - Scalers and IncrementalPCA match sklearn 1.8 to 1e-10.

- **Imputation (#56):**
  - New `MissingIndicator` (sklearn's, `features` = missing-only / all,
    `error_on_new`).
  - `SimpleImputer` gains sklearn's `add_indicator` and
    `keep_empty_features`, and `sample_weight` through
    `FitUnsupervisedWeighted` (weighted mean, median and mode; this is an
    extension, sklearn has no weights here).
  - **Breaking:** an all-missing column is dropped from the output, as in
    sklearn (its `fill_values` entry is NaN); before, `fit` returned an
    error. `keep_empty_features(true)` keeps it and imputes 0.

- **KnnImputer (#56):** a port of sklearn's `KNNImputer` with the
  `nan_euclidean` metric, `uniform` / `distance` weights, `add_indicator`
  and `keep_empty_features`. Rows to impute are processed in parallel.
  Matches sklearn 1.8 to 1e-12.

- **PolynomialFeatures (#56)** gains sklearn's `include_bias` and
  `degree=(min, max)` (`with_degree_range`), `powers()` and
  `feature_names_out()`. Each degree-k column is built from its degree-k−1
  prefix column times one feature, instead of recomputing every power.
  `degree = 0` with a bias is allowed, as in sklearn. Output matches
  sklearn 1.8.

- **TargetEncoder (#56):** a port of sklearn's `TargetEncoder`.
  - `smooth` is `Auto` (empirical Bayes) or a fixed value.
  - Continuous, binary and multiclass targets (one-vs-rest columns grouped
    by feature). Unknown categories map to the target mean; NaN is its own
    category.
  - `fit_transform` cross-fits (stratified folds for class targets).
  - New core hook `Fit::fit_transform_supervised`: `Supervised(encoder)` in
    a pipeline, FeatureUnion or ColumnTransformer gets the cross-fitted
    training output, as sklearn's pipeline does.
  - Encodings and `transform` match sklearn 1.8 to 1e-12. Fold assignment
    uses this crate's seeded splitters, so cross-fitted values differ from
    sklearn's.

- **RcsTransformer (#56):** restricted cubic spline expansion per feature.
  It wraps `anofox_regression::splines::Rcs` (`rms::rcs`): default knots
  at the RMS quantiles or explicit knots, `norm` options, rms-style
  column names. sklearn has no counterpart.

- **IterativeImputer (#56):** a port of sklearn's `IterativeImputer`.
  - Chained per-feature regressions; BayesianRidge by default, or any
    `ImputationEstimator` (`AnyRegressor` adapts a `Fit` + `Predict`
    regressor).
  - sklearn's early stopping, imputation orders, `skip_complete`, bounds,
    `n_nearest_features`, `add_indicator` and `keep_empty_features`.
  - `sample_posterior` draws from the truncated predictive normal.
  - `transform` replays the fitted sequence.
  - With `sample_posterior = false` it matches sklearn 1.8 (BayesianRidge
    to 1e-8, LinearRegression to 1e-9). Random draws use this crate's
    generator, so they differ from sklearn's. BayesianRidge's
    `predict_std` still has the centring deviation recorded in #72.
  - anofox-ml-preprocessing now depends on anofox-ml-regression.
- **AregImputer (#56):** multiple imputation through
  `anofox_regression::imputation::areg_impute` (`Hmisc::aregImpute`). It
  returns `n_impute` completed data sets to pool with Rubin's rules.

- **Python / facade (#56):**
  - New Python classes `IncrementalPCA`, `MissingIndicator`, `KNNImputer`,
    `IterativeImputer` (BayesianRidge estimator) and `TargetEncoder`, whose
    `fit_transform` returns the cross-fitted encoding.
  - `SimpleImputer` gains `add_indicator` / `keep_empty_features`, and
    `PolynomialFeatures` gains `include_bias` and a `degree` tuple.
  - `StandardScaler` and `SimpleImputer` accept `sample_weight`.
  - `partial_fit` is not exposed in Python yet.
  - The new Rust types are in the `anofox_ml` prelude.
  - `KnnImputer` breaks exact distance ties by training-row order. sklearn
    leaves the order unspecified (`argpartition`), so tied cases can differ.

- **Bagging with any base estimator (#50):** `BaggingRegressor::with_estimator(e)`
  and `BaggingClassifier::with_estimator(e)` bag any `Fit` + `Predict`
  estimator, like sklearn's `estimator=` parameter. They keep the sampling
  options (rows, `max_features`, `bootstrap_features`, OOB score).
  - Each estimator is fitted on its drawn rows and columns.
    `fit_weighted` passes the drawn weights on to `FitWeighted` bases.
  - The classifier votes; `predict_proba` is the vote fraction, as sklearn
    does for bases without `predict_proba`.

- **Gradient boosting `warm_start` (#50):**
  `GradientBoostingRegressor::fit_warm_start(prev, X, y, w)` and the
  classifier's counterpart add trees to a fitted model until there are
  `n_estimators`. The earlier rounds' random draws are replayed, so
  without early stopping the result is identical to one fit with
  `n_estimators` trees, which is sklearn's guarantee. The early-stopping
  history restarts with each call, as in sklearn. Not exposed in Python
  yet.

- **f32 input for clustering / manifold / LOF (#61):**
  - Affected estimators: AgglomerativeClustering, AffinityPropagation,
    GaussianMixture, BayesianGaussianMixture, Birch, SpectralClustering,
    OPTICS, MeanShift, HDBSCAN, Isomap, LLE, ClassicalMds, t-SNE and
    LocalOutlierFactor.
  - They now implement `FitUnsupervised<F>` (and `Predict<F>` /
    `PredictProba<F>` where available) for any `F: Float`.
  - They compute in f64 and convert at the boundary (new core helpers
    `as_f64` / `from_f64`, zero-copy for f64). f32 data gives the same
    result as the same data in f64. Fitted attributes stay f64.

- **SVM persistence:** `Gamma` now round-trips through bincode, so fitted
  SVMs can be saved with `save_bincode` (they previously failed to load).
- More configs derive `Serialize` / `Deserialize` (regression, clustering,
  manifold, LDA/QDA, isolation forest, LOF, decompositions).

### Algorithms — performance

- **Agglomerative Ward** now uses Müllner's O(n²) nn-chain algorithm by
  default. Earlier attempts produced an inconsistent flat clustering because
  nn-chain processes merges in chain order rather than distance order. The
  fix runs the chain to a single cluster, sorts the recorded merges by
  distance ascending, and applies them via DSU until `n_clusters` remain.
  Set `RUSTML_AGGLO_NAIVE=1` to force the O(n³) sweep (regression-test
  cross-check).
