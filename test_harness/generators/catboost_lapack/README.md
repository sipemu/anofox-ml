# CatBoost CLAPACK golden data

`crates/anofox-ml-ensemble/src/catboost/lapack_golden.txt` holds bit patterns
of `ssygst(1, "L")` + `ssyev("V", "L")` (with catboost's `lwork`) as computed
by catboost 1.2.10's bundled f2c CLAPACK and CBLAS. The LDA calcer
(`embedding_features/lda.cpp`) runs exactly these two calls.

Regenerate against a catboost build tree `$B` (built from the v1.2.10 source
with `build_native.py --targets _catboost`):

```sh
clang -O3 -msse4.2 harness.c -o harness -Wl,--start-group \
  $B/contrib/libs/clapack/part1/liblibs-clapack-part1.a \
  $B/contrib/libs/clapack/part2/liblibs-clapack-part2.a \
  $B/contrib/libs/cblas/libcontrib-libs-cblas.a \
  $B/contrib/libs/libf2c/libcontrib-libs-libf2c.a -Wl,--end-group -lm
./harness consts   # slamch constants and ilaenv block sizes (all 1)
python gen_lapack_golden.py ../../../crates/anofox-ml-ensemble/src/catboost/lapack_golden.txt
```
