//! Exact parity with scikit-learn 1.8.0 for KMeans (fixed init),
//! GaussianMixture (fixed weights/means/precisions init) and
//! BayesianGaussianMixture (k-means init on data with a unique k-means
//! optimum), plus a statistical check of KMeans `n_init`.
//!
//! Reference values were produced with sklearn 1.8.0 / numpy; the generating
//! snippets are quoted next to each block.

// Data literals happen to contain values like 3.142 / 0.318.
#![allow(clippy::approx_constant)]

use anofox_ml_cluster::{
    BayesianGaussianMixture, CovarianceType, FittedBayesianGaussianMixture, GaussianMixture,
    KMeans, KMeansInit,
};
use anofox_ml_core::{FitUnsupervised, FitUnsupervisedWeighted, PredictProba};
use ndarray::{array, Array1, Array2};

// rng = np.random.RandomState(0)
// X = np.vstack([rng.randn(20, 2) * [1.0, 0.6] + [0, 0],
//                rng.randn(20, 2) * [0.8, 1.2] + [3, 2],
//                rng.randn(20, 2) * 0.7 + [0, 4]]).round(3)
const X: [[f64; 2]; 60] = [
    [1.764, 0.24],
    [0.979, 1.345],
    [1.868, -0.586],
    [0.95, -0.091],
    [-0.103, 0.246],
    [0.144, 0.873],
    [0.761, 0.073],
    [0.444, 0.2],
    [1.494, -0.123],
    [0.313, -0.512],
    [-2.553, 0.392],
    [0.864, -0.445],
    [2.27, -0.873],
    [0.046, -0.112],
    [1.533, 0.882],
    [0.155, 0.227],
    [-0.888, -1.188],
    [-0.348, 0.094],
    [1.23, 0.721],
    [-0.387, -0.181],
    [2.161, 0.296],
    [1.635, 4.341],
    [2.592, 1.474],
    [1.998, 2.933],
    [1.709, 1.745],
    [2.284, 2.464],
    [2.591, 0.583],
    [2.977, 2.514],
    [3.053, 2.363],
    [2.493, 1.565],
    [2.462, 1.569],
    [2.349, -0.072],
    [3.142, 1.518],
    [1.696, 2.555],
    [2.274, 2.062],
    [3.583, 2.155],
    [3.912, 0.518],
    [3.322, 1.178],
    [2.303, 1.305],
    [2.751, 2.067],
    [-0.816, 4.631],
    [0.326, 2.925],
    [1.042, 5.327],
    [0.825, 3.874],
    [-0.75, 4.738],
    [-0.282, 4.856],
    [0.146, 4.684],
    [0.249, 4.495],
    [0.007, 5.25],
    [0.089, 4.281],
    [1.318, 3.057],
    [-0.889, 4.679],
    [-0.821, 5.361],
    [-0.29, 3.477],
    [1.346, 5.036],
    [1.307, 4.634],
    [-0.603, 5.337],
    [-0.188, 4.562],
    [0.663, 3.891],
    [0.43, 4.646],
];

// r = np.random.RandomState(1)
// Y = np.vstack([r.randn(15, 2) * 0.6 + [0, 0], r.randn(15, 2) * [0.5, 0.9] + [3.0, 0],
//                r.randn(15, 2) * 0.7 + [0, 3.0]]).round(3)
const Y: [[f64; 2]; 45] = [
    [0.975, -0.367],
    [-0.317, -0.644],
    [0.519, -1.381],
    [1.047, -0.457],
    [0.191, -0.15],
    [0.877, -1.236],
    [-0.193, -0.23],
    [0.68, -0.66],
    [-0.103, -0.527],
    [0.025, 0.35],
    [-0.66, 0.687],
    [0.541, 0.301],
    [0.541, -0.41],
    [-0.074, -0.561],
    [-0.161, 0.318],
    [2.654, -0.357],
    [2.656, -0.761],
    [2.664, -0.011],
    [2.441, 0.211],
    [3.83, 0.668],
    [2.904, -0.799],
    [2.626, 1.523],
    [3.025, -0.573],
    [3.095, 1.89],
    [3.06, 0.555],
    [3.15, -0.317],
    [2.429, -0.314],
    [2.896, 0.528],
    [3.419, 0.838],
    [3.143, 0.797],
    [-0.528, 3.877],
    [0.359, 2.791],
    [0.342, 2.947],
    [0.792, 4.064],
    [1.53, 2.022],
    [-1.011, 2.647],
    [0.112, 3.613],
    [0.221, 1.584],
    [-0.214, 3.58],
    [0.161, 3.533],
    [-0.156, 2.859],
    [0.131, 3.287],
    [0.139, 3.083],
    [-0.469, 3.264],
    [0.085, 3.791],
];

// r = np.random.RandomState(3)
// cs = [[i * 4.0, j * 4.0] for i in range(4) for j in range(3)]
// Z = np.vstack([r.randn(10, 2) * 0.8 + c for c in cs]).round(3)
const Z: [[f64; 2]; 120] = [
    [1.431, 0.349],
    [0.077, -1.491],
    [-0.222, -0.284],
    [-0.066, -0.502],
    [-0.035, -0.382],
    [-1.051, 0.708],
    [0.705, 1.368],
    [0.04, -0.324],
    [-0.436, -1.237],
    [0.786, -0.881],
    [-0.948, 3.835],
    [1.189, 4.189],
    [-0.819, 3.43],
    [0.5, 3.872],
    [-0.615, 3.816],
    [0.596, 5.581],
    [-0.995, 3.499],
    [-0.643, 2.065],
    [-0.739, 3.181],
    [0.899, 3.894],
    [-1.299, 8.517],
    [-0.285, 6.605],
    [-0.477, 7.529],
    [-0.699, 8.024],
    [-1.799, 7.786],
    [0.811, 8.682],
    [0.887, 8.896],
    [1.19, 7.105],
    [0.677, 6.511],
    [-0.482, 6.468],
    [4.839, 1.067],
    [3.842, 1.42],
    [3.46, 0.12],
    [4.122, -0.851],
    [4.35, 1.551],
    [3.18, 0.719],
    [3.876, 1.416],
    [4.387, 0.541],
    [4.515, 0.199],
    [2.883, 1.113],
    [2.903, 4.191],
    [4.491, 3.33],
    [4.116, 4.934],
    [3.981, 3.289],
    [1.667, 3.223],
    [3.527, 3.587],
    [3.232, 4.302],
    [3.54, 3.912],
    [4.543, 3.316],
    [3.76, 5.727],
    [4.699, 6.965],
    [3.936, 8.452],
    [4.987, 8.119],
    [3.576, 7.416],
    [4.516, 8.25],
    [3.587, 7.849],
    [3.667, 8.58],
    [3.448, 8.389],
    [4.681, 8.389],
    [3.333, 9.076],
    [7.457, 0.341],
    [7.397, -1.395],
    [8.181, 0.23],
    [7.938, 0.221],
    [7.481, -0.59],
    [7.866, 1.527],
    [8.652, -0.416],
    [8.447, -0.383],
    [7.634, 0.687],
    [7.58, -1.341],
    [7.275, 4.071],
    [8.102, 4.993],
    [7.427, 4.585],
    [8.341, 3.881],
    [8.669, 4.394],
    [7.31, 4.857],
    [7.023, 4.048],
    [8.002, 4.34],
    [7.42, 3.972],
    [7.888, 4.798],
    [7.363, 8.058],
    [7.791, 6.962],
    [10.141, 7.943],
    [6.811, 9.127],
    [7.144, 8.297],
    [8.69, 7.481],
    [7.655, 7.568],
    [7.897, 6.702],
    [7.011, 7.887],
    [8.831, 8.505],
    [13.384, 0.555],
    [11.591, -0.098],
    [10.376, -0.769],
    [11.184, 0.216],
    [12.518, -0.448],
    [11.529, -1.237],
    [11.898, 0.199],
    [12.357, -0.626],
    [13.591, 0.956],
    [11.924, -0.422],
    [11.743, 4.121],
    [11.985, 4.387],
    [12.615, 5.093],
    [12.918, 3.912],
    [12.311, 3.69],
    [11.53, 5.529],
    [11.632, 5.593],
    [11.721, 4.202],
    [12.872, 4.019],
    [12.315, 3.807],
    [11.62, 7.867],
    [11.48, 9.305],
    [11.866, 9.378],
    [9.852, 8.015],
    [12.45, 7.765],
    [12.876, 8.512],
    [11.78, 8.348],
    [14.25, 8.202],
    [12.24, 7.648],
    [12.107, 6.969],
];

fn to2<const N: usize>(a: &[[f64; 2]; N]) -> Array2<f64> {
    Array2::from_shape_fn((N, 2), |(i, j)| a[i][j])
}

fn close2(a: &Array2<f64>, b: &Array2<f64>, tol: f64, what: &str) {
    assert_eq!(a.dim(), b.dim(), "{what}: shape");
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() <= tol, "{what}: {a} vs {b}");
    }
}

fn close1(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: len");
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() <= tol, "{what}: {a:?} vs {b:?}");
    }
}

// ───── KMeans ──────────────────────────────────────────────────────────────

fn x_rows(idx: &[usize]) -> Array2<f64> {
    Array2::from_shape_fn((idx.len(), 2), |(i, j)| X[idx[i]][j])
}

/// KMeans(3, init=X[[0, 1, 2]], n_init=1).fit(X)
#[test]
fn kmeans_fixed_init_matches_sklearn() {
    let x = to2(&X);
    let km = KMeans::new(3).with_init(KMeansInit::Array(x_rows(&[0, 1, 2])));
    let f = FitUnsupervised::<f64>::fit(&km, &x).unwrap();
    let want = array![
        [2.5592500000000005, 1.5836999999999999],
        [0.22590476190476216, 4.480095238095238],
        [0.47384210526315795, 0.015789473684210353]
    ];
    close2(f.centroids(), &want, 1e-12, "centers");
    assert!((f.inertia() - 71.93040525325814).abs() < 1e-10);
    assert_eq!(f.n_iter(), 6);
    let mut labels = vec![2.0; 20];
    labels[14] = 0.0;
    labels.push(0.0);
    labels.push(1.0);
    labels.extend(std::iter::repeat_n(0.0, 18));
    labels.extend(std::iter::repeat_n(1.0, 20));
    assert_eq!(f.labels().to_vec(), labels);
}

/// KMeans(3, init=[[0, 0], [3, 2], [100, 100]], n_init=1).fit(X): the third
/// centroid is empty after the first assignment and must be relocated to the
/// point farthest from its centroid.
#[test]
fn kmeans_empty_cluster_relocation_matches_sklearn() {
    let x = to2(&X);
    let init = array![[0.0, 0.0], [3.0, 2.0], [100.0, 100.0]];
    let km = KMeans::new(3).with_init(KMeansInit::Array(init));
    let f = FitUnsupervised::<f64>::fit(&km, &x).unwrap();
    let want = array![
        [0.683909090909091, 0.06390909090909114],
        [2.5811111111111114, 1.8680555555555556],
        [0.1713, 4.55125]
    ];
    close2(f.centroids(), &want, 1e-12, "centers");
    assert!((f.inertia() - 71.95033230858586).abs() < 1e-10);
    assert_eq!(f.n_iter(), 5);
    let mut labels = vec![0.0; 21];
    labels.push(2.0);
    labels.extend(std::iter::repeat_n(1.0, 9));
    labels.push(0.0);
    labels.extend(std::iter::repeat_n(1.0, 8));
    labels.extend(std::iter::repeat_n(2.0, 10));
    labels.push(1.0);
    labels.extend(std::iter::repeat_n(2.0, 9));
    assert_eq!(f.labels().to_vec(), labels);
}

/// KMeans(3, init=X[[0, 1, 2]]).fit(X, sample_weight=np.arange(60) % 3 + 1)
#[test]
fn kmeans_weighted_fixed_init_matches_sklearn() {
    let x = to2(&X);
    let w: Array1<f64> = (0..60).map(|i| (i % 3 + 1) as f64).collect();
    let km = KMeans::new(3).with_init(KMeansInit::Array(x_rows(&[0, 1, 2])));
    let f = km.fit_unsupervised_weighted(&x, Some(&w)).unwrap();
    let want = array![
        [2.453088888888889, 1.478822222222222],
        [0.1785, 4.352095238095238],
        [0.32918181818181813, -0.06975757575757591]
    ];
    close2(f.centroids(), &want, 1e-12, "centers");
    assert!((f.inertia() - 144.2325973109668).abs() < 1e-9);
    assert_eq!(f.n_iter(), 4);
}

/// KMeans(3, init=X[[0, 1, 2]], tol=0.5).fit(X): stops on the relative
/// tolerance `tol · mean(var(X))` after 3 iterations.
#[test]
fn kmeans_relative_tol_matches_sklearn() {
    let x = to2(&X);
    let km = KMeans::new(3)
        .with_tol(0.5)
        .with_init(KMeansInit::Array(x_rows(&[0, 1, 2])));
    let f = FitUnsupervised::<f64>::fit(&km, &x).unwrap();
    let want = array![
        [2.4512727272727277, 1.511772727272727],
        [0.22590476190476216, 4.480095238095238],
        [0.3682352941176472, -0.07558823529411818]
    ];
    close2(f.centroids(), &want, 1e-12, "centers");
    assert!((f.inertia() - 72.44190439974625).abs() < 1e-10);
    assert_eq!(f.n_iter(), 3);

    // Scale invariance: the tolerance is relative to the data variance, so
    // rescaling X (and the init) must not change the number of iterations.
    let s = 1000.0;
    let km_s = KMeans::new(3)
        .with_tol(0.5)
        .with_init(KMeansInit::Array(x_rows(&[0, 1, 2]) * s));
    let f_s = FitUnsupervised::<f64>::fit(&km_s, &(&x * s)).unwrap();
    assert_eq!(f_s.n_iter(), 3);
    close2(&(f_s.centroids() / s), &want, 1e-9, "scaled centers");
}

/// n_init behaviour, statistically, on 12 overlapping blobs (k = 12).
///
/// sklearn, 100 seeds, mean(inertia / best) with best = 127.92093083434344:
///   random   n_init=1: 1.643   random   n_init=10: 1.146
///   k-means++ n_init=1: 1.121  k-means++ n_init=10: 1.0007
#[test]
fn kmeans_n_init_statistics_match_sklearn() {
    let z = to2(&Z);
    let best = 127.92093083434344;
    let mean_ratio = |init: KMeansInit, n_init: Option<usize>| -> f64 {
        let mut s = 0.0;
        for seed in 0..100 {
            let mut km = KMeans::new(12).with_seed(seed).with_init(init.clone());
            km.n_init = n_init;
            let f = FitUnsupervised::<f64>::fit(&km, &z).unwrap();
            assert!(f.inertia() >= best * (1.0 - 1e-9), "beat the optimum?");
            s += f.inertia() / best;
        }
        s / 100.0
    };
    let r1 = mean_ratio(KMeansInit::Random, Some(1));
    let r10 = mean_ratio(KMeansInit::Random, None); // "auto" = 10 for random
    let p1 = mean_ratio(KMeansInit::KMeansPlusPlus, None); // "auto" = 1 for k-means++
    let p10 = mean_ratio(KMeansInit::KMeansPlusPlus, Some(10));
    eprintln!("random1={r1} random10={r10} kpp1={p1} kpp10={p10}");
    assert!((r1 - 1.643).abs() < 0.25, "random n_init=1: {r1}");
    assert!((r10 - 1.146).abs() < 0.08, "random n_init=10: {r10}");
    assert!((p1 - 1.121).abs() < 0.05, "k-means++ n_init=1: {p1}");
    assert!(p10 < 1.01, "k-means++ n_init=10: {p10}");
    assert!(r1 > r10 && p1 > p10 && r1 > p1);
}

#[test]
fn kmeans_n_init_auto_resolution() {
    assert_eq!(KMeans::new(3).effective_n_init(), 1);
    assert_eq!(
        KMeans::new(3)
            .with_init(KMeansInit::Random)
            .effective_n_init(),
        10
    );
    assert_eq!(
        KMeans::new(3)
            .with_n_init(5)
            .with_init(KMeansInit::Array(Array2::zeros((3, 2))))
            .effective_n_init(),
        1
    );
}

// ───── GaussianMixture ─────────────────────────────────────────────────────

struct GmmRef {
    ct: CovarianceType,
    precisions_init: Vec<Array2<f64>>,
    weights: [f64; 3],
    means: [[f64; 2]; 3],
    covariances: Vec<Array2<f64>>,
    precisions_cholesky: Vec<Array2<f64>>,
    lower_bound: f64,
    n_iter: usize,
    score: f64,
    bic: f64,
    aic: f64,
    score_samples3: [f64; 3],
    proba0: [f64; 3],
}

fn blocks_1xd(rows: &[[f64; 2]]) -> Vec<Array2<f64>> {
    rows.iter().map(|r| array![[r[0], r[1]]]).collect()
}

fn blocks_1x1(v: &[f64]) -> Vec<Array2<f64>> {
    v.iter().map(|&s| array![[s]]).collect()
}

/// GaussianMixture(3, covariance_type=ct, weights_init=[.3, .3, .4],
///                 means_init=[[.5, .5], [2.5, 2.5], [.5, 3.5]],
///                 precisions_init=P[ct]).fit(X)
fn gmm_refs() -> Vec<GmmRef> {
    vec![
        GmmRef {
            ct: CovarianceType::Full,
            precisions_init: vec![
                array![[1.0, 0.2], [0.2, 1.5]],
                array![[0.8, 0.0], [0.0, 0.8]],
                array![[2.0, -0.3], [-0.3, 1.0]],
            ],
            weights: [0.3903627147141472, 0.27481693896894077, 0.33482034631691204],
            means: [
                [0.8259733376483852, 0.10430004895104553],
                [2.5403376936902657, 1.960555297871572],
                [0.18410761886121363, 4.544003057686135],
            ],
            covariances: vec![
                array![
                    [1.5588495416707129, 0.08321523617398476],
                    [0.08321523617398476, 0.3445539574997734]
                ],
                array![
                    [0.45748658488353267, -0.2141961800290457],
                    [-0.21419618002904572, 0.4060695356692828]
                ],
                array![
                    [0.5870604003289249, -0.09545334676675137],
                    [-0.09545334676675137, 0.3886422448358223]
                ],
            ],
            precisions_cholesky: vec![
                array![
                    [0.8009361580068144, -0.09153513994682248],
                    [0.0, 1.7147041517077048]
                ],
                array![
                    [1.4784642183190675, 0.8466951869083528],
                    [0.0, 1.8083968138157294]
                ],
                array![
                    [1.3051445277846887, 0.26618462425907885],
                    [0.0, 1.637097675168898]
                ],
            ],
            lower_bound: -3.2100627135130417,
            n_iter: 8,
            score: -3.20984135989491,
            bic: 454.7848207451649,
            aic: 419.1809631873892,
            score_samples3: [-2.7535106524456796, -4.656831612052052, -3.627536047671344],
            proba0: [
                0.9992175546183618,
                0.0007824453190229993,
                6.261526486236963e-11,
            ],
        },
        GmmRef {
            ct: CovarianceType::Tied,
            precisions_init: vec![array![[1.2, 0.1], [0.1, 0.9]]],
            weights: [0.34853215434233736, 0.3195372417748998, 0.33193060388276274],
            means: [
                [0.6726356355439598, 0.011680426117599396],
                [2.453579860032131, 1.8259193569991359],
                [0.1920724126114088, 4.559426206902457],
            ],
            covariances: vec![array![
                [0.8883824192488354, -0.10084936580861924],
                [-0.10084936580861924, 0.384487006549207]
            ]],
            precisions_cholesky: vec![array![
                [1.0609624715675354, 0.18586445834909698],
                [0.0, 1.6372806694083648]
            ]],
            lower_bound: -3.2967578884341537,
            n_iter: 6,
            score: -3.2965181159183046,
            bic: 440.61996409463967,
            aic: 417.58217391019656,
            score_samples3: [-3.1375302222585866, -3.806349007870261, -3.429369979023053],
            proba0: [
                0.9619485398785387,
                0.03805146006656186,
                5.4899724742645366e-11,
            ],
        },
        GmmRef {
            ct: CovarianceType::Diag,
            precisions_init: blocks_1xd(&[[1.0, 1.5], [0.8, 0.8], [2.0, 1.0]]),
            weights: [0.35741964286486827, 0.3050283425386958, 0.33755201459643597],
            means: [
                [0.7649458592899407, 0.02692495670740749],
                [2.442156451838326, 1.8404791309541433],
                [0.18920458677622823, 4.532372671393832],
            ],
            covariances: blocks_1xd(&[
                [1.6589753508679042, 0.28584066968317884],
                [0.49321448304087384, 0.4734874983112268],
                [0.5868750479954726, 0.402719563934048],
            ]),
            precisions_cholesky: blocks_1xd(&[
                [0.7763901794180633, 1.870415055787531],
                [1.4239085228228037, 1.4532681034134738],
                [1.3053506129529608, 1.5757910737002596],
            ]),
            lower_bound: -3.2387654263689245,
            n_iter: 6,
            score: -3.238384576970485,
            bic: 445.9269731075676,
            aic: 416.60614923645824,
            score_samples3: [-2.8019283946288036, -4.362238882588102, -3.512694443703832],
            proba0: [
                0.9305581998225437,
                0.06944180015186277,
                2.5593617805624638e-11,
            ],
        },
        GmmRef {
            ct: CovarianceType::Spherical,
            precisions_init: blocks_1x1(&[1.0, 0.8, 1.5]),
            weights: [0.3499234123383066, 0.31127325632648495, 0.3388033313352085],
            means: [
                [0.6500931631731809, 0.08020396677310744],
                [2.5328458046525326, 1.7320953261013234],
                [0.19571832530761213, 4.52685376241515],
            ],
            covariances: blocks_1x1(&[0.8927454829715243, 0.5318171400134962, 0.500959252222184]),
            precisions_cholesky: blocks_1x1(&[
                1.0583667056740091,
                1.3712569284575766,
                1.4128589237196296,
            ]),
            lower_bound: -3.3370024670285092,
            n_iter: 5,
            score: -3.3369614313543368,
            bic: 445.4731619469635,
            aic: 422.4353717625204,
            score_samples3: [
                -3.2892337571574926,
                -3.4317417929496523,
                -3.8354106741324276,
            ],
            proba0: [0.823278925600589, 0.1767210717167919, 2.682619048358351e-09],
        },
    ]
}

#[test]
fn gmm_fixed_init_matches_sklearn_all_covariance_types() {
    let x = to2(&X);
    for r in gmm_refs() {
        let what = format!("{:?}", r.ct);
        let f = GaussianMixture::new(3)
            .with_covariance_type(r.ct)
            .with_weights_init(array![0.3, 0.3, 0.4])
            .with_means_init(array![[0.5, 0.5], [2.5, 2.5], [0.5, 3.5]])
            .with_precisions_init(r.precisions_init.clone())
            .fit(&x)
            .unwrap();
        let tol = 1e-6;
        close1(
            f.weights.as_slice().unwrap(),
            &r.weights,
            tol,
            &format!("{what} weights"),
        );
        close2(&f.means, &to2(&r.means), tol, &format!("{what} means"));
        assert_eq!(f.covariances.len(), r.covariances.len());
        for (a, b) in f.covariances.iter().zip(&r.covariances) {
            close2(a, b, tol, &format!("{what} cov"));
        }
        for (a, b) in f.precisions_cholesky.iter().zip(&r.precisions_cholesky) {
            close2(a, b, tol, &format!("{what} prec_chol"));
        }
        assert!(
            (f.lower_bound - r.lower_bound).abs() < tol,
            "{what} lb {}",
            f.lower_bound
        );
        assert_eq!(f.n_iter(), r.n_iter, "{what} n_iter");
        assert!(f.converged(), "{what} converged");
        // Final E-step: the stored log-likelihood is score(X).
        assert!((f.log_likelihood - r.score).abs() < tol, "{what} ll");
        assert!((f.score(&x).unwrap() - r.score).abs() < tol, "{what} score");
        assert!((f.bic(&x).unwrap() - r.bic).abs() < 1e-4, "{what} bic");
        assert!((f.aic(&x).unwrap() - r.aic).abs() < 1e-4, "{what} aic");
        let ss = f.score_samples(&x).unwrap();
        close1(
            &ss.as_slice().unwrap()[..3],
            &r.score_samples3,
            tol,
            &format!("{what} ss"),
        );
        let p = f
            .predict_proba(&x.slice(ndarray::s![0..1, ..]).to_owned())
            .unwrap();
        close1(
            p.row(0).as_slice().unwrap(),
            &r.proba0,
            tol,
            &format!("{what} proba"),
        );
    }
}

// ───── BayesianGaussianMixture ─────────────────────────────────────────────

struct BgmmRef {
    ct: CovarianceType,
    lower_bound: f64,
    n_iter: usize,
    weights: [f64; 3],
    means: [[f64; 2]; 3],
    covariances: Vec<Array2<f64>>,
    dof: [f64; 3],
    wc: [f64; 3],
    proba0: [f64; 3],
}

/// BayesianGaussianMixture(3, covariance_type=ct, tol=1e-6, max_iter=500,
///     weight_concentration_prior_type="dirichlet_distribution").fit(Y)
/// Components sorted by (means[:, 0], means[:, 1]). Y has a unique k-means
/// optimum (same partition for every k-means seed), so the k-means
/// initialisation is implementation-independent.
fn bgmm_refs() -> Vec<BgmmRef> {
    vec![
        BgmmRef {
            ct: CovarianceType::Full,
            lower_bound: -79.18697830807237,
            n_iter: 8,
            weights: [
                0.32232461412771607,
                0.33705580791713313,
                0.34061957795515074,
            ],
            means: [
                [0.13577315461861345, 3.033085444753316],
                [0.3096041884700584, -0.21941707121154674],
                [2.788787085402102, 0.33453603422583894],
            ],
            covariances: vec![
                array![
                    [0.4332641048022256, -0.2260687264232873],
                    [-0.2260687264232873, 0.7865304375424192]
                ],
                array![
                    [0.3744708749494224, -0.10867632811932385],
                    [-0.10867632811932385, 0.5778842480632087]
                ],
                array![
                    [0.4500536462810931, -0.08025970581562307],
                    [-0.08025970581562308, 0.7726329275803724]
                ],
            ],
            dof: [16.493598916541607, 17.171233830854796, 17.335167252603604],
            wc: [14.826932249874943, 15.50456716418813, 15.668500585936938],
            proba0: [
                0.0008870001253940041,
                0.9811083505529579,
                0.0180046493216483,
            ],
        },
        BgmmRef {
            ct: CovarianceType::Tied,
            lower_bound: -79.77558298275684,
            n_iter: 6,
            weights: [0.3172895655579122, 0.3403324416168409, 0.34237799282524695],
            means: [
                [0.12435646751694077, 3.0506094192958324],
                [0.31099952901036904, -0.20594442979219044],
                [2.78334554432583, 0.3480082822993901],
            ],
            covariances: vec![array![
                [0.4170126451698771, -0.13940782542366403],
                [-0.13940782542366403, 0.7199873187104061]
            ]],
            dof: [17.000000000000007; 3],
            wc: [14.595320015663965, 15.655292314374686, 15.749387669961363],
            proba0: [
                0.00044737920092990165,
                0.9903248387190718,
                0.009227782079998076,
            ],
        },
        BgmmRef {
            ct: CovarianceType::Diag,
            lower_bound: -81.04801348068467,
            n_iter: 9,
            weights: [0.3226890492384952, 0.3339223445015085, 0.34338860625999634],
            means: [
                [0.12109253313992967, 3.0218978886440477],
                [0.31605776364300914, -0.22228038245379256],
                [2.7773306013494548, 0.3402659310369211],
            ],
            covariances: blocks_1xd(&[
                [0.4226566964805253, 0.8196540674778765],
                [0.37439473350764774, 0.5808940529527157],
                [0.4633531880367256, 0.7828710089087866],
            ]),
            dof: [16.510362931637445, 17.027094513736056, 17.4625425546265],
            wc: [14.843696264970777, 15.36042784706939, 15.795875887959832],
            proba0: [
                0.0005128688633771281,
                0.9687269608254416,
                0.030760170311181237,
            ],
        },
        BgmmRef {
            ct: CovarianceType::Spherical,
            lower_bound: -82.7461956229776,
            n_iter: 8,
            weights: [0.3243733905325169, 0.33297466024893757, 0.3426519492185456],
            means: [
                [0.13750059008569668, 3.0249511681851846],
                [0.3129958434757592, -0.222121848429347],
                [2.7707374318747187, 0.3230750329276367],
            ],
            covariances: blocks_1x1(&[0.6150912999140028, 0.48020321447649184, 0.6268789224385706]),
            dof: [16.587842631162445, 16.983501038117794, 17.428656330719765],
            wc: [14.92117596449578, 15.31683437145113, 15.761989664053099],
            proba0: [
                5.613582157408066e-05,
                0.9373536645860288,
                0.06259019959239709,
            ],
        },
    ]
}

fn sorted_order(f: &FittedBayesianGaussianMixture) -> Vec<usize> {
    let mut o: Vec<usize> = (0..f.means.nrows()).collect();
    o.sort_by(|&a, &b| {
        (f.means[[a, 0]], f.means[[a, 1]])
            .partial_cmp(&(f.means[[b, 0]], f.means[[b, 1]]))
            .unwrap()
    });
    o
}

#[test]
fn bgmm_elbo_and_posterior_match_sklearn_all_covariance_types() {
    let y = to2(&Y);
    for r in bgmm_refs() {
        let what = format!("{:?}", r.ct);
        let f = BayesianGaussianMixture::new(3)
            .with_covariance_type(r.ct)
            .with_tol(1e-6)
            .with_max_iter(500)
            .fit(&y)
            .unwrap();
        let o = sorted_order(&f);
        let tol = 1e-6;
        assert!(
            (f.lower_bound - r.lower_bound).abs() < tol,
            "{what} lower bound {} vs {}",
            f.lower_bound,
            r.lower_bound
        );
        assert_eq!(f.n_iter, r.n_iter, "{what} n_iter");
        assert!(f.converged(), "{what} converged");
        let pick = |a: &Array1<f64>| o.iter().map(|&c| a[c]).collect::<Vec<_>>();
        close1(
            &pick(&f.weights),
            &r.weights,
            tol,
            &format!("{what} weights"),
        );
        close1(&pick(&f.dof_posterior), &r.dof, tol, &format!("{what} dof"));
        close1(
            &pick(&f.weight_concentration),
            &r.wc,
            tol,
            &format!("{what} wc"),
        );
        let means = Array2::from_shape_fn((3, 2), |(i, j)| f.means[[o[i], j]]);
        close2(&means, &to2(&r.means), tol, &format!("{what} means"));
        if r.ct == CovarianceType::Tied {
            close2(
                &f.covariances[0],
                &r.covariances[0],
                tol,
                &format!("{what} cov"),
            );
        } else {
            for (i, &c) in o.iter().enumerate() {
                close2(
                    &f.covariances[c],
                    &r.covariances[i],
                    tol,
                    &format!("{what} cov"),
                );
            }
        }
        let p = f
            .predict_proba(&y.slice(ndarray::s![0..1, ..]).to_owned())
            .unwrap();
        let p0: Vec<f64> = o.iter().map(|&c| p[[0, c]]).collect();
        close1(&p0, &r.proba0, tol, &format!("{what} proba"));
    }
}
