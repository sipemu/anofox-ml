//! numpy's legacy `RandomState` (MT19937) and the scikit-learn splits built
//! on it, reproduced draw for draw.
//!
//! Estimators that must match scikit-learn exactly where it draws random
//! numbers (e.g. the validation split of HistGradientBoosting's early
//! stopping) use these instead of `rand`.

use crate::{Result, RustMlError};

const N: usize = 624;
const M: usize = 397;

/// numpy's `np.random.RandomState` seeded with an integer.
#[derive(Debug, Clone)]
pub struct RandomState {
    key: [u32; N],
    pos: usize,
}

impl RandomState {
    /// `np.random.RandomState(seed)` (MT19937's `init_genrand`).
    pub fn new(seed: u32) -> Self {
        let mut key = [0u32; N];
        key[0] = seed;
        for i in 1..N {
            key[i] = 1_812_433_253u32
                .wrapping_mul(key[i - 1] ^ (key[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { key, pos: N }
    }

    fn generate(&mut self) {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;
        for i in 0..N {
            let y = (self.key[i] & UPPER) | (self.key[(i + 1) % N] & LOWER);
            let mag = if y & 1 == 1 { MATRIX_A } else { 0 };
            self.key[i] = self.key[(i + M) % N] ^ (y >> 1) ^ mag;
        }
        self.pos = 0;
    }

    /// The next 32-bit output (`genrand_int32`).
    pub fn next_u32(&mut self) -> u32 {
        if self.pos == N {
            self.generate();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// A uniform integer in `[0, max]` by masked rejection (numpy's
    /// `random_interval`, used by `shuffle`), for `max < 2^32`.
    pub fn interval(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        let mask = u32::MAX >> max.leading_zeros();
        loop {
            let v = self.next_u32() & mask;
            if v <= max {
                return v;
            }
        }
    }

    /// `randint(high)` for `1 <= high <= 2^32` (masked rejection, as the
    /// legacy bounded-integer generator draws it).
    pub fn randint(&mut self, high: u64) -> u64 {
        assert!((1..=1 << 32).contains(&high), "randint high out of range");
        u64::from(self.interval((high - 1) as u32))
    }

    /// `shuffle(values)` (Fisher–Yates from the end).
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.interval(i as u32) as usize;
            values.swap(i, j);
        }
    }

    /// `permutation(n)`.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        self.shuffle(&mut p);
        p
    }

    /// `choice(values, size, replace=False)` without probabilities.
    pub fn choice_without_replacement<T: Copy>(&mut self, values: &[T], size: usize) -> Vec<T> {
        self.permutation(values.len())
            .into_iter()
            .take(size)
            .map(|i| values[i])
            .collect()
    }
}

/// sklearn's `_validate_shuffle_split` for a test size and no train size:
/// `(n_train, n_test)`.
pub fn shuffle_split_sizes(n_samples: usize, test_size: TestSize) -> Result<(usize, usize)> {
    let n_test = match test_size {
        TestSize::Fraction(f) if f > 0.0 && f < 1.0 => (f * n_samples as f64).ceil() as usize,
        TestSize::Count(c) if c > 0 && c < n_samples => c,
        other => {
            return Err(RustMlError::InvalidParameter(format!(
                "test size {other:?} is invalid for {n_samples} samples"
            )))
        }
    };
    let n_train = n_samples - n_test;
    if n_train == 0 || n_test == 0 {
        return Err(RustMlError::InvalidParameter(format!(
            "with {n_samples} samples and test size {test_size:?} one side of the split is empty"
        )));
    }
    Ok((n_train, n_test))
}

/// Size of the held-out side of a split (sklearn's float or int
/// `test_size`).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TestSize {
    /// Fraction in (0, 1); the count is rounded up.
    Fraction(f64),
    /// Absolute number of samples.
    Count(usize),
}

/// `train_test_split(..., test_size, random_state=seed)` without
/// stratification (sklearn's `ShuffleSplit`): `(train, test)` row indices in
/// sklearn's order.
pub fn train_test_split_indices(
    n_samples: usize,
    test_size: TestSize,
    seed: u32,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let (n_train, n_test) = shuffle_split_sizes(n_samples, test_size)?;
    let perm = RandomState::new(seed).permutation(n_samples);
    Ok((
        perm[n_test..n_test + n_train].to_vec(),
        perm[..n_test].to_vec(),
    ))
}

/// sklearn's `_approximate_mode`: the most likely class counts when drawing
/// `n_draws` samples without replacement, ties broken at random.
fn approximate_mode(class_counts: &[usize], n_draws: usize, rng: &mut RandomState) -> Vec<usize> {
    let total: usize = class_counts.iter().sum();
    let continuous: Vec<f64> = class_counts
        .iter()
        .map(|&c| c as f64 / total as f64 * n_draws as f64)
        .collect();
    let mut floored: Vec<f64> = continuous.iter().map(|c| c.floor()).collect();
    let mut need = n_draws as i64 - floored.iter().sum::<f64>() as i64;
    if need > 0 {
        let remainder: Vec<f64> = continuous
            .iter()
            .zip(&floored)
            .map(|(c, f)| c - f)
            .collect();
        let mut values = remainder.clone();
        values.sort_by(|a, b| b.total_cmp(a));
        values.dedup();
        for value in values {
            let inds: Vec<usize> = (0..remainder.len())
                .filter(|&i| remainder[i] == value)
                .collect();
            let add_now = inds.len().min(need as usize);
            for i in rng.choice_without_replacement(&inds, add_now) {
                floored[i] += 1.0;
            }
            need -= add_now as i64;
            if need == 0 {
                break;
            }
        }
    }
    floored.iter().map(|&f| f as usize).collect()
}

/// `train_test_split(..., test_size, stratify=labels, random_state=seed)`
/// (sklearn's `StratifiedShuffleSplit`): `(train, test)` row indices in
/// sklearn's order. `labels` are class indices `0..n_classes`.
pub fn stratified_train_test_split_indices(
    labels: &[usize],
    test_size: TestSize,
    seed: u32,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let n_samples = labels.len();
    let (n_train, n_test) = shuffle_split_sizes(n_samples, test_size)?;
    let n_classes = labels.iter().max().map_or(0, |&m| m + 1);
    let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (i, &l) in labels.iter().enumerate() {
        class_indices[l].push(i);
    }
    let class_counts: Vec<usize> = class_indices.iter().map(Vec::len).collect();
    if class_counts.iter().any(|&c| c < 2) {
        return Err(RustMlError::InvalidParameter(
            "the least populated class has only 1 member, which is too few to stratify".into(),
        ));
    }
    if n_train < n_classes || n_test < n_classes {
        return Err(RustMlError::InvalidParameter(format!(
            "train ({n_train}) and test ({n_test}) sizes must be at least the number of classes \
             ({n_classes})"
        )));
    }
    let mut rng = RandomState::new(seed);
    let n_i = approximate_mode(&class_counts, n_train, &mut rng);
    let remaining: Vec<usize> = class_counts.iter().zip(&n_i).map(|(c, n)| c - n).collect();
    let t_i = approximate_mode(&remaining, n_test, &mut rng);
    let (mut train, mut test) = (Vec::with_capacity(n_train), Vec::with_capacity(n_test));
    for (k, members) in class_indices.iter().enumerate() {
        let perm = rng.permutation(members.len());
        let shuffled: Vec<usize> = perm.iter().map(|&p| members[p]).collect();
        train.extend_from_slice(&shuffled[..n_i[k]]);
        test.extend_from_slice(&shuffled[n_i[k]..n_i[k] + t_i[k]]);
    }
    rng.shuffle(&mut train);
    rng.shuffle(&mut test);
    Ok((train, test))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_numpy() {
        // np.random.RandomState(5489).randint(2**32, dtype=np.uint64, size=3)
        // and RandomState(42) permutation / randint(2**32 - 1, dtype="u8"),
        // numpy 2.x.
        let mut rs = RandomState::new(5489);
        let draws: Vec<u64> = (0..3).map(|_| rs.randint(1 << 32)).collect();
        assert_eq!(draws, vec![3_499_211_612, 581_869_302, 3_890_346_734]);
        assert_eq!(
            RandomState::new(42).permutation(10),
            vec![8, 1, 5, 0, 7, 2, 9, 4, 3, 6]
        );
        assert_eq!(RandomState::new(42).randint(u32::MAX as u64), 1_608_637_542);
    }

    #[test]
    fn splits_match_sklearn() {
        // sklearn 1.8 train_test_split(np.arange(n), ...) (numpy 2.5).
        let split = |n, t, seed| train_test_split_indices(n, t, seed).unwrap();
        let (train, test) = split(20, TestSize::Fraction(0.25), 3);
        assert_eq!(test, vec![14, 2, 1, 17, 4]);
        assert_eq!(
            train,
            vec![16, 6, 7, 15, 12, 9, 11, 19, 18, 13, 5, 0, 8, 3, 10]
        );
        let (train, test) = split(30, TestSize::Count(7), 4_000_000_000);
        assert_eq!(test, vec![2, 8, 4, 13, 7, 26, 9]);
        assert_eq!(
            train,
            vec![
                16, 6, 14, 28, 24, 22, 19, 15, 17, 11, 20, 0, 23, 25, 21, 1, 3, 5, 18, 29, 27, 10,
                12
            ]
        );

        // stratify=[0]*12 + [1]*8, and three classes with a tie in
        // `_approximate_mode`: [0]*7 + [1]*7 + [2]*9 + [1]*3.
        let labels: Vec<usize> = (0..20).map(|i| usize::from(i >= 12)).collect();
        let (train, test) =
            stratified_train_test_split_indices(&labels, TestSize::Fraction(0.25), 3).unwrap();
        assert_eq!(test, vec![17, 9, 10, 14, 8]);
        assert_eq!(
            train,
            vec![15, 5, 6, 0, 2, 13, 11, 16, 1, 7, 18, 4, 19, 3, 12]
        );
        let labels: Vec<usize> = [vec![0; 7], vec![1; 7], vec![2; 9], vec![1; 3]].concat();
        let (train, test) =
            stratified_train_test_split_indices(&labels, TestSize::Fraction(0.3), 11).unwrap();
        assert_eq!(test, vec![9, 1, 8, 23, 22, 18, 0, 16]);
        assert_eq!(
            train,
            vec![24, 3, 12, 13, 11, 14, 21, 5, 19, 25, 17, 2, 20, 4, 7, 10, 6, 15]
        );
    }
}
