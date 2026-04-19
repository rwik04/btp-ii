import numpy as np

from lfgd.pca_axis import estimate_bias_axis, estimate_bias_axis_with_sign


def test_estimate_bias_axis_shapes():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(20, 16))
    axis, mu, centered = estimate_bias_axis(embeddings)
    assert axis.shape == (16,)
    assert mu.shape == (16,)
    assert centered.shape == (20, 16)
    assert np.isclose(np.linalg.norm(axis), 1.0, atol=1e-6)


def test_resolved_axis_projects_first_half_positive():
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(20, 8))
    embeddings[:10] += 2.0
    axis = estimate_bias_axis_with_sign(embeddings)
    centered = embeddings - embeddings.mean(axis=0)
    projection = np.dot(centered[:10].mean(axis=0), axis)
    assert projection >= 0
