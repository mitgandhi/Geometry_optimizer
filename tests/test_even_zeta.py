import os
import sys
import random
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import constraints
np = pytest.importorskip("numpy")
import nsga3_algorithm as nsga
import bayesian_optimization as bo

param_bounds = {
    'dK': {'min': 19.0, 'max': 20.0},
    'dZ': {'min': 19.2, 'max': 20.5},
    'lK': {'min': 50.0, 'max': 70.0},
    'lF': {'min': 30.0, 'max': 40.0},
    'zeta': {'min': 3, 'max': 7}
}


def test_generate_valid_parameters_even_zeta():
    cm = constraints.ConstraintManager()
    params = cm.generate_valid_parameters(param_bounds, max_attempts=100)
    assert params is not None
    assert params[4] % 2 == 0


def test_simple_generate_individual_even_zeta():
    alg = object.__new__(nsga.SimpleNSGA3)
    alg.param_bounds = param_bounds
    alg.constraint_manager = constraints.ConstraintManager()
    individual = alg.generate_individual()
    assert individual[4] % 2 == 0


def test_advanced_generate_individual_even_zeta():
    alg = object.__new__(nsga.AdvancedNSGA3)
    alg.param_bounds = param_bounds
    alg.constraint_manager = constraints.ConstraintManager()
    alg.n_partitions = 4
    individual = alg.generate_individual()
    assert individual[4] % 2 == 0
def test_bo_generate_individual_even_zeta():
    alg = object.__new__(bo.BayesianOptimization)
    alg.param_bounds = param_bounds
    alg.param_names = ['dK', 'dZ', 'lK', 'lF', 'zeta']
    alg.constraint_manager = constraints.ConstraintManager()
    individual = alg.generate_individual()
    assert individual[4] % 2 == 0


def test_polynomial_mutation_even_zeta(monkeypatch):
    alg = object.__new__(nsga.SimpleNSGA3)
    alg.param_bounds = param_bounds
    alg.constraint_manager = constraints.ConstraintManager()

    def fake_random():
        return 0.0

    monkeypatch.setattr(random, "random", fake_random)
    mutated = alg.polynomial_mutation([19.2, 19.8, 60.0, 35.0, 5])
    assert mutated[4] % 2 == 0


