import numpy as np
import pytest

from flyres.subgraph import select_subgraph
from flyres.synthetic import synthetic_connectome


@pytest.fixture(scope="session")
def conn():
    return synthetic_connectome(n=800, mean_degree=20, seed=0)


@pytest.fixture(scope="session")
def sub(conn):
    return select_subgraph(conn, n_neurons=300, n_inputs=20, input_filter={"superclass": "sensory"}, verbose=False)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
