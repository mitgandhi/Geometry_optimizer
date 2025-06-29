import pytest

np = pytest.importorskip("numpy")
import nsga3_algorithm as nsga


def test_domination_sort_single_front():
    obj = [[1, 2]]
    simple = object.__new__(nsga.SimpleNSGA3)
    advanced = object.__new__(nsga.AdvancedNSGA3)
    assert simple.domination_sort(obj) == [[0]]
    assert advanced.domination_sort(obj) == [[0]]

