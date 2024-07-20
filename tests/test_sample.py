import pytest
from mlb_simulator.data.data_utils import query_mlb_db


def inc(x):
    return x + 1


def test_answer():
    assert inc(2) == 3
