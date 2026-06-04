# AffinityPropagation — sklearn parity

Issue: [#16](https://github.com/sipemu/rustml/issues/16) (partial — HDBSCAN / OPTICS / Birch pending)

## What

Affinity Propagation in `rustml-cluster::affinity_propagation`. Uses
similarity `s_{i,k} = -||x_i - x_k||²`, message-passes responsibilities and
availabilities until exemplars stabilise.

## Reference

`sklearn.cluster.AffinityPropagation` — sklearn 1.8.0.

## Validation

Unit test asserts that on a 2-blob fixture with a deliberately-set preference,
AP recovers ≥ 2 exemplars and assigns points correctly. A full sklearn golden
test is not provided because AP is notoriously sensitive to preference and
the median-of-similarities default differs in edge cases.

## Not yet implemented

- **HDBSCAN**, **OPTICS**, **Birch** — still pending in #16.
- `affinity='precomputed'` / arbitrary similarity matrices.
- `predict` on new points.
