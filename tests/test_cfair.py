import numpy as np
import pytest

from metrics.cfair import compute_cfair_at_k, compute_cfair_score


def test_cfair_perfectly_balanced_list_is_near_one():
    k = 6
    lean = np.array([-1 + (2 * j - 1) / k for j in range(1, k + 1)], dtype=float)
    rel = np.ones(k, dtype=float)
    score = compute_cfair_score(lean.tolist(), rel.tolist(), k=k)
    assert np.isclose(score, 1.0, atol=1e-6)


def test_cfair_at_k_rejects_length_mismatch():
    with pytest.raises(ValueError):
        compute_cfair_at_k([0.1, -0.1], [0.9], 2)
