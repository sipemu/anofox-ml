//! CatBoost's KNN embedding calcer (`embedding_features/knn.cpp`): an online
//! HNSW index (`library/cpp/online_hnsw`) over the rows seen so far, built
//! row by row. Its searches use `std::priority_queue` (libc++'s heap order
//! decides ties) and the SSE L2 distance (four lanes).

use super::{HeapKey, StdPriorityQueue};
use std::collections::{HashSet, VecDeque};

/// CatBoost's `SearchNeighborhoodSize` for KNN.
const SEARCH: usize = 300;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Neighbor {
    dist: f32,
    id: usize,
}

/// A max-queue entry (`TNeighborMaxQueue`: farthest on top).
#[derive(Clone, Copy)]
struct Far(Neighbor);

impl HeapKey for Far {
    fn key(&self) -> f64 {
        f64::from(self.0.dist)
    }
}

/// A min-queue entry (`TNeighborMinQueue`: nearest on top).
#[derive(Clone, Copy)]
struct Near(Neighbor);

impl HeapKey for Near {
    fn key(&self) -> f64 {
        -f64::from(self.0.dist)
    }
}

/// `L2SqrDistance(const float*, const float*, int)` (SSE): four lane sums,
/// the tail into lane 0, then the lanes in order.
pub(crate) fn l2_sqr(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 4];
    let full = a.len() / 4 * 4;
    for c in (0..full).step_by(4) {
        for k in 0..4 {
            let d = a[c + k] - b[c + k];
            lanes[k] += d * d;
        }
    }
    for i in full..a.len() {
        let d = b[i] - a[i];
        lanes[0] += d * d;
    }
    lanes[0] + lanes[1] + lanes[2] + lanes[3]
}

/// `TDynamicDenseGraph`: per vertex `max_nc` neighbor slots, of which the
/// first `nc` are in use (padding is id 0 at distance 0).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Level {
    max_nc: usize,
    max_size: usize,
    size: usize,
    nc: usize,
    dists: Vec<f32>,
    ids: Vec<usize>,
}

impl Level {
    fn new(max_nc: usize, max_size: usize) -> Self {
        Self {
            max_nc,
            max_size,
            size: 0,
            nc: 0,
            dists: Vec::new(),
            ids: Vec::new(),
        }
    }

    /// A copy of `other` with `max_nc` slots per vertex.
    fn widened(max_nc: usize, max_size: usize, other: &Level) -> Self {
        let mut l = Self::new(max_nc, max_size);
        l.size = other.size;
        l.nc = other.nc;
        if max_nc == other.max_nc {
            l.dists = other.dists.clone();
            l.ids = other.ids.clone();
            return l;
        }
        for v in 0..other.size {
            l.dists.extend_from_slice(other.dists_of(v));
            l.dists.resize(l.dists.len() + max_nc - l.nc, 0.0);
            l.ids.extend_from_slice(other.ids_of(v));
            l.ids.resize(l.ids.len() + max_nc - l.nc, 0);
        }
        l
    }

    fn ids_of(&self, v: usize) -> &[usize] {
        &self.ids[v * self.max_nc..v * self.max_nc + self.nc]
    }

    fn dists_of(&self, v: usize) -> &[f32] {
        &self.dists[v * self.max_nc..v * self.max_nc + self.nc]
    }

    fn append(&mut self, neighbors: &[Neighbor]) {
        debug_assert!(self.size < self.max_size && neighbors.len() <= self.max_nc);
        for n in neighbors {
            self.dists.push(n.dist);
            self.ids.push(n.id);
        }
        self.dists
            .resize(self.dists.len() + self.max_nc - neighbors.len(), 0.0);
        self.ids
            .resize(self.ids.len() + self.max_nc - neighbors.len(), 0);
        self.size += 1;
        if self.nc < self.max_nc {
            self.nc = self.size - 1;
        }
    }

    fn replace(&mut self, v: usize, neighbors: &[Neighbor]) {
        let at = v * self.max_nc;
        for (p, n) in neighbors.iter().enumerate() {
            self.dists[at + p] = n.dist;
            self.ids[at + p] = n.id;
        }
    }
}

/// `TOnlineHnswDenseVectorIndex<float, TL2SqrDistance>` with CatBoost's
/// options (`MaxNeighbors` = k, `SearchNeighborhoodSize` = 300).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct OnlineHnsw {
    dim: usize,
    max_neighbors: usize,
    decay: usize,
    /// Front: the bottom level (every item); back: the top one.
    levels: VecDeque<Level>,
    level_sizes: VecDeque<usize>,
    diverse: Vec<usize>,
    data: Vec<f32>,
}

impl OnlineHnsw {
    pub(crate) fn new(dim: usize, max_neighbors: usize) -> Self {
        let decay = (max_neighbors / 2).max(2);
        let mut levels = VecDeque::new();
        levels.push_front(Level::new((decay - 1).min(max_neighbors), decay));
        Self {
            dim,
            max_neighbors,
            decay,
            levels,
            level_sizes: VecDeque::from([decay]),
            diverse: Vec::new(),
            data: Vec::new(),
        }
    }

    fn n_items(&self) -> usize {
        self.data.len() / self.dim.max(1)
    }

    fn item(&self, id: usize) -> &[f32] {
        &self.data[id * self.dim..(id + 1) * self.dim]
    }

    fn dist(&self, query: &[f32], id: usize) -> f32 {
        l2_sqr(query, self.item(id))
    }

    /// `GetNearestNeighbors`: at most `top` neighbors, nearest first.
    pub(crate) fn nearest(&self, query: &[f32], top: usize) -> Vec<Neighbor> {
        if self.max_neighbors + 1 >= self.n_items() {
            return self.naive(query, top);
        }
        let mut r = self.approximate(query, top);
        r.reverse();
        r
    }

    fn naive(&self, query: &[f32], top: usize) -> Vec<Neighbor> {
        let mut nearest: StdPriorityQueue<Far> = StdPriorityQueue(Vec::new());
        for id in 0..self.n_items() {
            let dist = self.dist(query, id);
            let full = nearest.0.len() == top;
            if !full || dist < nearest.top().0.dist {
                nearest.push(Far(Neighbor { dist, id }));
                if full {
                    nearest.pop();
                }
            }
        }
        let mut out = Vec::with_capacity(nearest.0.len());
        while !nearest.0.is_empty() {
            out.push(nearest.pop().0);
        }
        out.reverse();
        out
    }

    /// `NRoutines::FindApproximateNeighbors`: farthest first.
    fn approximate(&self, query: &[f32], top: usize) -> Vec<Neighbor> {
        let mut entry = 0usize;
        let mut entry_dist = self.dist(query, entry);
        for level in (1..self.levels.len()).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                for &id in self.levels[level].ids_of(entry) {
                    let d = self.dist(query, id);
                    if d < entry_dist {
                        entry_dist = d;
                        entry = id;
                        changed = true;
                    }
                }
            }
        }
        let start = Neighbor {
            dist: entry_dist,
            id: entry,
        };
        let mut nearest = StdPriorityQueue(vec![Far(start)]);
        let mut candidates = StdPriorityQueue(vec![Near(start)]);
        let mut visited = HashSet::from([entry]);
        let bottom = &self.levels[0];
        while !candidates.0.is_empty() {
            let cur = candidates.pop().0;
            if nearest.top().0.dist < cur.dist {
                break;
            }
            for &id in bottom.ids_of(cur.id) {
                if visited.contains(&id) {
                    continue;
                }
                let dist = self.dist(query, id);
                if nearest.0.len() < SEARCH || dist < nearest.top().0.dist {
                    nearest.push(Far(Neighbor { dist, id }));
                    candidates.push(Near(Neighbor { dist, id }));
                    visited.insert(id);
                    if nearest.0.len() > SEARCH {
                        nearest.pop();
                    }
                }
            }
        }
        while nearest.0.len() > top {
            nearest.pop();
        }
        let mut out = Vec::with_capacity(nearest.0.len());
        while !nearest.0.is_empty() {
            out.push(nearest.pop().0);
        }
        out
    }

    /// `GetNearestNeighborsAndAddItem`.
    pub(crate) fn add(&mut self, item: &[f32]) {
        let neighbors = self.nearest(item, usize::MAX);
        self.data.extend_from_slice(item);
        self.add_level_if_full();
        self.extend_last_level(&neighbors);
    }

    fn add_level_if_full(&mut self) {
        let l = self.levels.len();
        let last_max = self.level_sizes[self.level_sizes.len() - l];
        if self.levels[0].size == last_max {
            if self.level_sizes.len() == l {
                let next = self.level_sizes[0] * self.decay;
                self.level_sizes.push_front(next);
            }
            let size = self.level_sizes[self.level_sizes.len() - 1 - l];
            let level = Level::widened(self.max_neighbors.min(size - 1), size, &self.levels[0]);
            self.levels.push_front(level);
        }
    }

    fn extend_last_level(&mut self, neighbors: &[Neighbor]) {
        let (trimmed, n_diverse) = self.trim(neighbors);
        self.diverse.push(n_diverse);
        let new_id = self.levels[0].size;
        for &edge in &trimmed {
            self.try_add_inverse_edge(edge, new_id);
        }
        self.levels[0].append(&trimmed);
    }

    fn try_add_inverse_edge(&mut self, edge: Neighbor, source: usize) {
        let level = &self.levels[0];
        let new_dist = edge.dist;
        let dest = edge.id;
        let n_diverse = self.diverse[dest];
        let new_count = (level.nc + 1).min(self.max_neighbors);
        let ids = level.ids_of(dest);
        let dists = level.dists_of(dest);
        let mut is_diverse = true;
        let mut pos = 0;
        while pos < n_diverse {
            if new_dist < dists[pos] {
                break;
            }
            let d = l2_sqr(self.item(ids[pos]), self.item(source));
            if d < new_dist {
                is_diverse = false;
                break;
            }
            pos += 1;
        }
        if level.nc > 0 && is_diverse && pos < n_diverse {
            self.retrim_and_add_inverse_edge(edge, source);
            return;
        }
        let mut new_pos = n_diverse;
        if !is_diverse {
            while new_pos < level.nc && dists[new_pos] < new_dist {
                new_pos += 1;
            }
        }
        if new_pos >= new_count {
            return;
        }
        self.diverse[dest] += usize::from(is_diverse);
        // AddEdgeOnPosition.
        let level = &self.levels[0];
        let (ids, dists) = (level.ids_of(dest), level.dists_of(dest));
        let mut out = Vec::with_capacity(new_count);
        for p in 0..new_pos {
            out.push(Neighbor {
                dist: dists[p],
                id: ids[p],
            });
        }
        out.push(Neighbor {
            dist: new_dist,
            id: source,
        });
        let mut old = new_pos;
        while out.len() < new_count {
            out.push(Neighbor {
                dist: dists[old],
                id: ids[old],
            });
            old += 1;
        }
        self.levels[0].replace(dest, &out);
    }

    fn retrim_and_add_inverse_edge(&mut self, edge: Neighbor, source: usize) {
        let level = &self.levels[0];
        let new_dist = edge.dist;
        let dest = edge.id;
        let n_diverse = self.diverse[dest];
        let (ids, dists) = (level.ids_of(dest), level.dists_of(dest));
        let total = level.nc + 1;
        let mut merged = Vec::with_capacity(total);
        let mut new_available = true;
        let (mut dpos, mut cpos) = (0, n_diverse);
        while merged.len() < total {
            let diverse_available = dpos != n_diverse;
            let clustering_available = cpos != level.nc;
            let mut next = Neighbor { dist: 0.0, id: 0 };
            if clustering_available && (!diverse_available || dists[cpos] < dists[dpos]) {
                next = Neighbor {
                    dist: dists[cpos],
                    id: ids[cpos],
                };
                cpos += 1;
            } else if diverse_available {
                next = Neighbor {
                    dist: dists[dpos],
                    id: ids[dpos],
                };
                dpos += 1;
            }
            let add_old = diverse_available || clustering_available;
            if new_available && (!add_old || new_dist < next.dist) {
                merged.push(Neighbor {
                    dist: new_dist,
                    id: source,
                });
                new_available = false;
            }
            if add_old && merged.len() < total {
                merged.push(next);
            }
        }
        let (trimmed, n) = self.trim(&merged);
        self.diverse[dest] = n;
        self.levels[0].replace(dest, &trimmed);
    }

    /// `TrimSortedNeighbors`: the diverse neighbors first, then the closest
    /// others; and the number of diverse ones.
    fn trim(&self, neighbors: &[Neighbor]) -> (Vec<Neighbor>, usize) {
        if neighbors.is_empty() {
            return (Vec::new(), 0);
        }
        let max = self.max_neighbors.min(neighbors.len());
        let mut result = vec![neighbors[0]];
        let mut clustering = Vec::new();
        for current in &neighbors[1..] {
            if result.len() >= max {
                break;
            }
            let item = self.item(current.id);
            let take = result
                .iter()
                .all(|t| l2_sqr(item, self.item(t.id)) >= current.dist);
            if take {
                result.push(*current);
            } else if result.len() + clustering.len() < max {
                clustering.push(*current);
            }
        }
        let n_diverse = result.len();
        let mut c = 0;
        while result.len() < max {
            result.push(clustering[c]);
            c += 1;
        }
        (result, n_diverse)
    }
}

/// CatBoost's `TKNNCalcer` with its visitor: the class counts (or the mean
/// target) of the `k` nearest rows seen so far.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct Knn {
    k: usize,
    /// Classes, or `None` for regression.
    n_classes: Option<usize>,
    index: OnlineHnsw,
    targets: Vec<f32>,
}

impl Knn {
    pub(crate) fn new(dim: usize, n_classes: Option<usize>, k: usize) -> Self {
        Self {
            k,
            n_classes,
            index: OnlineHnsw::new(dim, k),
            targets: Vec::new(),
        }
    }

    pub(crate) fn dim(&self) -> usize {
        self.index.dim
    }

    pub(crate) fn n_features(n_classes: Option<usize>) -> usize {
        n_classes.unwrap_or(1)
    }

    fn features(&self, neighbors: &[Neighbor]) -> Vec<f32> {
        let mut out = vec![0.0f32; Self::n_features(self.n_classes)];
        if self.n_classes.is_some() {
            for n in neighbors {
                out[self.targets[n.id] as usize] += 1.0;
            }
        } else if !neighbors.is_empty() {
            for n in neighbors {
                out[0] += self.targets[n.id];
            }
            out[0] /= neighbors.len() as f32;
        }
        out
    }

    pub(crate) fn compute(&self, x: &[f32]) -> Vec<f32> {
        self.features(&self.index.nearest(x, self.k))
    }

    /// `target` is the class (classification) or the target.
    pub(crate) fn update(&mut self, target: f32, x: &[f32]) {
        self.index.add(x);
        self.targets.push(target);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_uses_four_lanes() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [0.5f32, 0.0, 1.0, 1.0, 2.0, 2.0];
        // The tail (4 and 5) goes into lane 0.
        let want = (0.25 + 9.0 + 16.0) + 4.0 + 4.0 + 9.0;
        assert_eq!(l2_sqr(&a, &b), want);
    }

    /// Few rows: an exact search; many: the HNSW finds the nearest of a
    /// well-separated query.
    #[test]
    fn knn_counts_the_nearest_classes() {
        let mut knn = Knn::new(2, Some(2), 3);
        assert_eq!(knn.compute(&[0.0, 0.0]), vec![0.0, 0.0]);
        for i in 0..200 {
            let c = i % 2;
            let x = [
                c as f32 * 10.0 + (i as f32 * 0.37).sin(),
                (i as f32 * 0.11).cos(),
            ];
            knn.update(c as f32, &x);
        }
        assert_eq!(knn.compute(&[10.0, 0.0]), vec![0.0, 3.0]);
        assert_eq!(knn.compute(&[0.0, 0.0]), vec![3.0, 0.0]);
    }
}
