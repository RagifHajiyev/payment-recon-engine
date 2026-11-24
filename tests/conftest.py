import pytest

from recon_engine import generate_data as g


@pytest.fixture(scope="session")
def frames():
    """20k synthetic transactions, shared across tests."""
    return g.generate(g.GenConfig(n_txn=20_000, seed=7))
