use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rustml_core::Float;

/// A KD-tree for efficient nearest neighbor search.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct KdTree<F: Float> {
    nodes: Vec<KdNode<F>>,
    n_dims: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
struct KdNode<F: Float> {
    point: Vec<F>,
    index: usize,
    left: Option<usize>,
    right: Option<usize>,
    split_dim: usize,
}

/// A neighbor entry for the max-heap. Ordered by distance descending
/// so the farthest neighbor is at the top of the heap.
struct HeapEntry<F: Float> {
    dist_sq: F,
    index: usize,
}

impl<F: Float> PartialEq for HeapEntry<F> {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq == other.dist_sq && self.index == other.index
    }
}

impl<F: Float> Eq for HeapEntry<F> {}

impl<F: Float> PartialOrd for HeapEntry<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: Float> Ord for HeapEntry<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by distance, then by index (higher index = "worse" = on top)
        // so that when we evict, we keep lower-indexed entries for tie-breaking.
        self.dist_sq
            .partial_cmp(&other.dist_sq)
            .unwrap_or(Ordering::Equal)
            .then(self.index.cmp(&other.index))
    }
}

impl<F: Float> KdTree<F> {
    /// Build a KD-tree from a set of points.
    ///
    /// `points` is a slice of (point_data, original_index) pairs.
    pub fn build(points: &[(Vec<F>, usize)], n_dims: usize) -> Self {
        let mut tree = KdTree {
            nodes: Vec::with_capacity(points.len()),
            n_dims,
        };

        if !points.is_empty() {
            let mut indices: Vec<usize> = (0..points.len()).collect();
            tree.build_recursive(points, &mut indices, 0);
        }

        tree
    }

    fn build_recursive(
        &mut self,
        points: &[(Vec<F>, usize)],
        indices: &mut [usize],
        depth: usize,
    ) -> Option<usize> {
        if indices.is_empty() {
            return None;
        }

        let split_dim = depth % self.n_dims;

        // Sort by split dimension
        indices.sort_by(|&a, &b| {
            points[a].0[split_dim]
                .partial_cmp(&points[b].0[split_dim])
                .unwrap()
        });

        let median = indices.len() / 2;
        let median_idx = indices[median];

        let node_idx = self.nodes.len();
        self.nodes.push(KdNode {
            point: points[median_idx].0.clone(),
            index: points[median_idx].1,
            left: None,
            right: None,
            split_dim,
        });

        let (left_slice, right_slice) = indices.split_at_mut(median);
        let right_slice = &mut right_slice[1..]; // skip median

        let left = self.build_recursive(points, left_slice, depth + 1);
        let right = self.build_recursive(points, right_slice, depth + 1);

        self.nodes[node_idx].left = left;
        self.nodes[node_idx].right = right;

        Some(node_idx)
    }

    /// Find the k nearest neighbors to `query`.
    ///
    /// Returns a vector of (distance, original_index) sorted by (distance, index) ascending.
    /// Ties at the boundary are broken by preferring lower indices (matching brute-force).
    pub fn query_k_nearest(&self, query: &[F], k: usize) -> Vec<(F, usize)> {
        let mut heap: BinaryHeap<HeapEntry<F>> = BinaryHeap::with_capacity(k + 1);

        if !self.nodes.is_empty() {
            self.search_heap(0, query, k, &mut heap);
        }

        // Drain the heap into a vec, then sort by (distance, index) ascending
        let mut result: Vec<(F, usize)> = heap
            .into_iter()
            .map(|e| (e.dist_sq.sqrt(), e.index))
            .collect();
        result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        result.truncate(k);
        result
    }

    /// Search using a bounded max-heap of size k.
    /// The heap top is always the farthest of the current k-best,
    /// giving O(1) pruning threshold lookup.
    fn search_heap(
        &self,
        node_idx: usize,
        query: &[F],
        k: usize,
        heap: &mut BinaryHeap<HeapEntry<F>>,
    ) {
        let node = &self.nodes[node_idx];
        let dist_sq = squared_distance(&node.point, query);

        if heap.len() < k {
            heap.push(HeapEntry {
                dist_sq,
                index: node.index,
            });
        } else if let Some(worst) = heap.peek() {
            // Replace if this point is better than the worst in our k-best,
            // or if it's the same distance but has a lower index (tie-breaking).
            if dist_sq < worst.dist_sq || (dist_sq == worst.dist_sq && node.index < worst.index) {
                heap.pop();
                heap.push(HeapEntry {
                    dist_sq,
                    index: node.index,
                });
            }
        }

        let diff = query[node.split_dim] - node.point[node.split_dim];
        let diff_sq = diff * diff;

        // Visit nearer subtree first
        let (near, far) = if diff <= F::zero() {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        if let Some(near_idx) = near {
            self.search_heap(near_idx, query, k, heap);
        }

        // Prune: only visit far subtree if the splitting plane is closer than
        // the kth-nearest distance found so far (or we have < k candidates)
        let should_visit_far = if heap.len() < k {
            true
        } else {
            diff_sq <= heap.peek().unwrap().dist_sq
        };

        if should_visit_far {
            if let Some(far_idx) = far {
                self.search_heap(far_idx, query, k, heap);
            }
        }
    }
}

#[inline]
fn squared_distance<F: Float>(a: &[F], b: &[F]) -> F {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc0 = F::zero();
    let mut acc1 = F::zero();
    let mut acc2 = F::zero();
    let mut acc3 = F::zero();

    let mut i = 0;
    for _ in 0..chunks {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        acc0 += d0 * d0;
        acc1 += d1 * d1;
        acc2 += d2 * d2;
        acc3 += d3 * d3;
        i += 4;
    }

    for j in 0..remainder {
        let d = a[i + j] - b[i + j];
        acc0 += d * d;
    }

    (acc0 + acc1) + (acc2 + acc3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kdtree_basic() {
        let points: Vec<(Vec<f64>, usize)> = vec![
            (vec![0.0, 0.0], 0),
            (vec![1.0, 0.0], 1),
            (vec![0.0, 1.0], 2),
            (vec![1.0, 1.0], 3),
            (vec![10.0, 10.0], 4),
        ];

        let tree = KdTree::build(&points, 2);

        // Query near (0,0)
        let result = tree.query_k_nearest(&[0.1, 0.1], 3);
        assert_eq!(result.len(), 3);
        // Nearest should be index 0 (0,0)
        assert_eq!(result[0].1, 0);
    }

    #[test]
    fn test_kdtree_single_point() {
        let points = vec![(vec![5.0, 5.0], 0)];
        let tree = KdTree::build(&points, 2);
        let result = tree.query_k_nearest(&[0.0, 0.0], 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 0);
    }

    #[test]
    fn test_kdtree_all_points() {
        let points: Vec<(Vec<f64>, usize)> = vec![(vec![0.0], 0), (vec![1.0], 1), (vec![2.0], 2)];
        let tree = KdTree::build(&points, 1);
        let result = tree.query_k_nearest(&[0.5], 3);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_kdtree_tie_breaking() {
        // Two points equidistant from query: lower index should come first
        let points: Vec<(Vec<f64>, usize)> = vec![(vec![1.0, 0.0], 0), (vec![-1.0, 0.0], 1)];
        let tree = KdTree::build(&points, 2);
        let result = tree.query_k_nearest(&[0.0, 0.0], 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 0); // index 0 wins the tie
    }

    #[test]
    fn test_kdtree_matches_brute_force() {
        let points: Vec<(Vec<f64>, usize)> = vec![
            (vec![2.0, 3.0], 0),
            (vec![5.0, 4.0], 1),
            (vec![9.0, 6.0], 2),
            (vec![4.0, 7.0], 3),
            (vec![8.0, 1.0], 4),
            (vec![7.0, 2.0], 5),
        ];
        let tree = KdTree::build(&points, 2);
        let query = [5.0, 5.0];
        let k = 3;

        // Brute force
        let mut dists: Vec<(f64, usize)> = points
            .iter()
            .map(|(p, idx)| {
                let d: f64 = p
                    .iter()
                    .zip(query.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                (d, *idx)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        let brute: Vec<usize> = dists.iter().take(k).map(|&(_, idx)| idx).collect();

        let kd_result: Vec<usize> = tree
            .query_k_nearest(&query, k)
            .iter()
            .map(|&(_, idx)| idx)
            .collect();
        assert_eq!(kd_result, brute);
    }
}
