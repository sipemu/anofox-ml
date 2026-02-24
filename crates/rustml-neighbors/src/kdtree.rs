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

struct Neighbor<F: Float> {
    dist_sq: F,
    index: usize,
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
        // Use a larger heap to capture tie candidates, then sort deterministically
        let mut candidates: Vec<Neighbor<F>> = Vec::new();

        if !self.nodes.is_empty() {
            self.search_collecting(0, query, k, &mut candidates);
        }

        // Sort by (distance, index) to match brute-force tie-breaking
        candidates.sort_by(|a, b| {
            a.dist_sq
                .partial_cmp(&b.dist_sq)
                .unwrap()
                .then(a.index.cmp(&b.index))
        });

        candidates
            .into_iter()
            .take(k)
            .map(|n| (n.dist_sq.sqrt(), n.index))
            .collect()
    }

    /// Search collecting candidates. Uses KD-tree pruning but collects
    /// all points within the kth-nearest distance (including ties).
    fn search_collecting(
        &self,
        node_idx: usize,
        query: &[F],
        k: usize,
        candidates: &mut Vec<Neighbor<F>>,
    ) {
        let node = &self.nodes[node_idx];
        let dist_sq = squared_distance(&node.point, query);

        candidates.push(Neighbor {
            dist_sq,
            index: node.index,
        });

        let diff = query[node.split_dim] - node.point[node.split_dim];
        let diff_sq = diff * diff;

        // Visit nearer subtree first
        let (near, far) = if diff <= F::zero() {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        if let Some(near_idx) = near {
            self.search_collecting(near_idx, query, k, candidates);
        }

        // Prune: only visit far subtree if the splitting plane is closer than
        // the kth-nearest distance found so far
        let should_visit_far = if candidates.len() < k {
            true
        } else {
            // Find kth-smallest distance so far
            let mut dists: Vec<F> = candidates.iter().map(|n| n.dist_sq).collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            diff_sq <= dists[k - 1]
        };

        if should_visit_far {
            if let Some(far_idx) = far {
                self.search_collecting(far_idx, query, k, candidates);
            }
        }
    }

}

fn squared_distance<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .fold(F::zero(), |acc, v| acc + v)
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
        let points: Vec<(Vec<f64>, usize)> = vec![
            (vec![0.0], 0),
            (vec![1.0], 1),
            (vec![2.0], 2),
        ];
        let tree = KdTree::build(&points, 1);
        let result = tree.query_k_nearest(&[0.5], 3);
        assert_eq!(result.len(), 3);
    }
}
