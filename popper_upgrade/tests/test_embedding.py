import numpy as np

from popper_upgrade.skills.embedding import Embedder
from popper_upgrade.tests.fakes import FakeEmbedder


def test_fake_embedder_deterministic():
    fake = FakeEmbedder()

    assert fake.encode("text") == fake.encode("text")
    assert fake.encode("text") != fake.encode("other")
    assert len(fake.encode("text")) == 384
    assert abs(np.linalg.norm(fake.encode("text")) - 1.0) < 1e-9


def test_embedder_dim_is_384():
    embedder = Embedder()

    vector = embedder.encode("gender bias in occupations")

    assert embedder.model_name == "all-MiniLM-L6-v2"
    assert embedder.dim == 384
    assert len(vector) == 384
    assert abs(np.linalg.norm(vector) - 1.0) < 1e-5
