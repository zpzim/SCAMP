"""Unit tests for the unified pyscamp.join() API.

Covers (1) the validation special-cases -- the combinatorial argument checks
that are easy to get wrong -- and (2) numerical agreement between join() and
the existing named functions, including the index=False and left/right paths
that join() adds.

Runs two ways:
    python test_join_api.py      # plain script, no pytest needed (CI path)
    pytest test_join_api.py      # if pytest is available
"""

import sys

import numpy as np

import pyscamp as mp
from pyscamp import join, JoinResult, KNNMatches

M = 128
_rng = np.random.RandomState(1234)
A = _rng.normal(size=3000)
B = _rng.normal(size=2400)

# 1NN profiles store correlation (higher is better). With pearson=True the API
# returns correlations, so the "best of two directions" reduction is max.
ATOL = 2e-4


def _expect_error(exc, fn):
    try:
        fn()
    except exc:
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, no error raised")


# --------------------------------------------------------------------------
# Validation special-cases (no compute)
# --------------------------------------------------------------------------

def test_requires_window():
    _expect_error(TypeError, lambda: join(A))


def test_unknown_method():
    _expect_error(ValueError, lambda: join(A, m=M, method="bogus"))


def test_k_only_for_knn():
    _expect_error(ValueError, lambda: join(A, m=M, k=5))


def test_knn_requires_k():
    _expect_error(ValueError, lambda: join(A, m=M, method="knn"))


def test_matrix_dims_only_for_matrix():
    # even the value that equals the matrix default must be rejected off-method
    _expect_error(ValueError, lambda: join(A, m=M, mheight=50))
    _expect_error(ValueError, lambda: join(A, m=M, mwidth=10))


def test_index_only_for_1nn():
    _expect_error(ValueError, lambda: join(A, m=M, method="sum", index=False))
    _expect_error(ValueError, lambda: join(A, m=M, method="matrix", index=False))


def test_threshold_not_for_1nn():
    _expect_error(ValueError, lambda: join(A, m=M, threshold=0.5))


def test_left_right_not_for_matrix():
    _expect_error(ValueError, lambda: join(A, m=M, method="matrix", left_right=True))


def test_trivial_match_is_abjoin_only():
    _expect_error(ValueError, lambda: join(A, m=M, allow_trivial_match=False))


# --------------------------------------------------------------------------
# Shape / field-population behavior
# --------------------------------------------------------------------------

def test_1nn_self_populates_profile_and_index():
    r = join(A, m=M, pearson=True)
    assert isinstance(r, JoinResult) and r.method == "1nn"
    assert r.profile is not None and r.index is not None
    assert r.profile.shape == (len(A) - M + 1,)
    assert r.left_profile is None and r.matrix is None and r.matches is None


def test_1nn_index_false_drops_index():
    r = join(A, m=M, index=False, pearson=True)
    assert r.profile is not None and r.index is None


def test_matrix_default_and_explicit_dims():
    assert join(A, m=M, method="matrix").matrix.shape == (50, 50)
    assert join(A, m=M, method="matrix", mheight=20, mwidth=30).matrix.shape == (20, 30)


def test_knn_returns_arrays():
    r = join(A, m=M, method="knn", k=3, pearson=True)
    assert isinstance(r.matches, KNNMatches)
    c, rr, d = r.matches
    assert c.shape == rr.shape == d.shape
    assert np.issubdtype(c.dtype, np.integer)
    assert np.issubdtype(d.dtype, np.floating)


def test_left_right_populates_both_directions():
    r = join(A, B, m=M, left_right=True, pearson=True)
    for f in ("left_profile", "left_index", "right_profile", "right_index"):
        assert getattr(r, f) is not None, f
    assert r.profile is None and r.index is None


# --------------------------------------------------------------------------
# Numerical agreement with the existing named functions
# --------------------------------------------------------------------------

def test_1nn_self_matches_selfjoin():
    r = join(A, m=M, pearson=True)
    prof, idx = mp.selfjoin(A, M, pearson=True)
    assert np.allclose(r.profile, prof, atol=ATOL, equal_nan=True)
    assert np.array_equal(r.index, idx)


def test_1nn_ab_matches_abjoin():
    r = join(A, B, m=M, pearson=True)
    prof, idx = mp.abjoin(A, B, M, pearson=True)
    assert np.allclose(r.profile, prof, atol=ATOL, equal_nan=True)
    assert np.array_equal(r.index, idx)


def test_index_false_profile_matches_indexed():
    with_idx = join(A, m=M, pearson=True).profile
    without = join(A, m=M, index=False, pearson=True).profile
    assert np.allclose(with_idx, without, atol=ATOL, equal_nan=True)


def test_sum_matches_selfjoin_sum():
    r = join(A, m=M, method="sum", threshold=0.5)
    ref = mp.selfjoin_sum(A, M, threshold=0.5)
    assert np.allclose(r.profile, ref, atol=1e-6, equal_nan=True)


def test_matrix_matches_selfjoin_matrix():
    r = join(A, m=M, method="matrix", mheight=25, mwidth=25, pearson=True)
    ref = mp.selfjoin_matrix(A, M, mheight=25, mwidth=25, pearson=True)
    assert np.allclose(r.matrix, ref, atol=ATOL, equal_nan=True)


def test_ab_left_right_matches_swapped_abjoins():
    r = join(A, B, m=M, left_right=True, pearson=True)
    lp, li = mp.abjoin(A, B, M, pearson=True)   # left  = A-in-B
    rp, ri = mp.abjoin(B, A, M, pearson=True)   # right = B-in-A
    assert np.allclose(r.left_profile, lp, atol=ATOL, equal_nan=True)
    assert np.array_equal(r.left_index, li)
    assert np.allclose(r.right_profile, rp, atol=ATOL, equal_nan=True)
    assert np.array_equal(r.right_index, ri)


def test_self_left_right_combines_to_full_profile():
    # Combined self-join profile == elementwise best (max corr) of the left
    # (preceding) and right (subsequent) profiles. fmax ignores the no-match
    # NaNs at the series ends.
    full = join(A, m=M, pearson=True).profile
    lr = join(A, m=M, left_right=True, pearson=True)
    combined = np.fmax(lr.left_profile, lr.right_profile)
    assert np.allclose(full, combined, atol=ATOL, equal_nan=True)


def test_knn_arrays_match_named_knn():
    r = join(A, m=M, method="knn", k=4, threshold=0.0, pearson=True)
    ref = mp.selfjoin_knn(A, M, 4, threshold=0.0, pearson=True)  # list[(col,row,dist)]
    assert len(r.matches.cols) == len(ref)
    got = set(zip(r.matches.cols.tolist(), r.matches.rows.tolist()))
    exp = set((int(c), int(rr)) for c, rr, _ in ref)
    assert got == exp


# --------------------------------------------------------------------------
# Plain-script runner (so CI can `python test_join_api.py` without pytest)
# --------------------------------------------------------------------------

def _run_all():
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[ OK ] {name}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed.")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
