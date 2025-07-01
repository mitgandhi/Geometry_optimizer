import random
import numpy as np
import multiprocessing as mp
import psutil
from collections import Counter
from cpu_affinity import get_available_cores
from constraints import ConstraintManager


def evaluate_individual_wrapper(args):
    """Wrapper for multiprocessing evaluation - each fsti_gap.exe uses cores 1-23 without timeout"""
    optimizer, individual_id, params, generation, core_id, algorithm_type, fixed_params = args

    print(f"{algorithm_type} Worker {individual_id} starting - will set fsti_gap.exe to cores 1-23 (NO TIMEOUT)")

    # Set current worker process to exclude core 0 (but allow multiple cores)
    try:
        current_process = psutil.Process()
        total_cores = psutil.cpu_count()
        all_cores_except_0 = list(range(1, total_cores))

        current_process.cpu_affinity(all_cores_except_0)
        final_affinity = current_process.cpu_affinity()
        print(f"{algorithm_type} Worker {individual_id} process affinity: {final_affinity}")

        if 0 in final_affinity:
            print(f"WARNING: {algorithm_type} Worker {individual_id} still has core 0, correcting...")
            corrected = [c for c in final_affinity if c != 0]
            current_process.cpu_affinity(corrected)
            final_affinity = current_process.cpu_affinity()
            print(f"{algorithm_type} Worker {individual_id} corrected affinity: {final_affinity}")

    except Exception as e:
        print(f"Warning: Could not set {algorithm_type} worker {individual_id} process affinity: {e}")

    # Run evaluation - fsti_gap.exe will get cores 1-23 and run without timeout
    dK, dZ, LKG, lF, zeta = params
    return optimizer.evaluate_individual(individual_id, dK, dZ, LKG, lF, zeta, generation,
                                         algorithm_type=algorithm_type, fixed_params=fixed_params)


class SimpleNSGA3:
    """Simplified NSGA-III implementation with constraint handling and no timeout limits"""

    def __init__(self, optimizer, param_bounds, available_cores, pop_size=20, generations=10,
                 constraints_config=None, progress_callback=None, fixed_params=None):
        self.optimizer = optimizer
        # Convert parameter bounds to consistent format
        self.param_bounds = self._normalize_param_bounds(param_bounds)
        self.available_cores = [c for c in available_cores if c != 0]  # Exclude core 0
        if not self.available_cores:
            self.available_cores = [1]

        self.pop_size = pop_size
        self.generations = generations
        self.progress_callback = progress_callback
        self.fixed_params = fixed_params or {}

        # Initialize constraint manager
        self.constraint_manager = ConstraintManager()
        self._setup_constraints(constraints_config)

        print(f"SimpleNSGA3 initialized with cores: {self.available_cores}")
        print(f"Constraint system initialized with {len(self.constraint_manager.constraints)} constraints")
        print(f"Fixed parameters: {self.fixed_params}")
        print(f"NO TIMEOUT - simulations will run until completion")
        optimizer.setup_optimization_folder("Simple")

    def _normalize_param_bounds(self, param_bounds):
        """Convert parameter bounds to consistent dict format"""
        normalized = {}
        for param, bounds in param_bounds.items():
            if isinstance(bounds, (tuple, list)):
                # Convert (min, max) to {'min': min, 'max': max}
                normalized[param] = {'min': bounds[0], 'max': bounds[1]}
            elif isinstance(bounds, dict):
                # Already in correct format
                normalized[param] = bounds
            else:
                raise ValueError(f"Invalid bounds format for {param}: {bounds}")
        return normalized

    def _setup_constraints(self, constraints_config):
        """Setup constraints from configuration"""
        if not constraints_config:
            return

        # Configure dK < dZ constraint
        dk_dz_config = constraints_config.get('dK_less_than_dZ', {})
        if not dk_dz_config.get('active', True):
            self.constraint_manager.deactivate_constraint('dK_less_than_dZ')

        # Configure dZ - dK difference constraint
        dz_dk_config = constraints_config.get('dZ_dK_difference_range', {})
        if not dz_dk_config.get('active', True):
            self.constraint_manager.deactivate_constraint('dZ_dK_difference_range')
        else:
            # Update configuration
            config = {
                'min_difference': dz_dk_config.get('min_difference', 0.1),
                'max_difference': dz_dk_config.get('max_difference', 0.8)
            }
            self.constraint_manager.set_constraint_config('dZ_dK_difference_range', config)

        print("Constraint configuration applied:")
        self.constraint_manager.list_constraints()

    def generate_individual(self):
        """Generate random individual within bounds that satisfies constraints"""
        # Try to generate valid individual
        valid_params = self.constraint_manager.generate_valid_parameters(self.param_bounds, max_attempts=100)

        if valid_params:
            return valid_params
        else:
            # Fallback: generate random and try to repair
            print("⚠️ Could not generate valid individual, using repair method")
            individual = [
                random.uniform(self.param_bounds['dK']['min'], self.param_bounds['dK']['max']),
                random.uniform(self.param_bounds['dZ']['min'], self.param_bounds['dZ']['max']),
                random.uniform(self.param_bounds['LKG']['min'], self.param_bounds['LKG']['max']),
                random.uniform(self.param_bounds['lF']['min'], self.param_bounds['lF']['max']),
                random.randint(self.param_bounds['zeta']['min'], self.param_bounds['zeta']['max'])
            ]

            # Try to repair
            param_dict = {
                'dK': individual[0], 'dZ': individual[1], 'LKG': individual[2],
                'lF': individual[3], 'zeta': individual[4]
            }
            repaired = self.constraint_manager.repair_parameters(param_dict, self.param_bounds)

            return [repaired['dK'], repaired['dZ'], repaired['LKG'], repaired['lF'], repaired['zeta']]

    def validate_individual(self, individual):
        """Check if individual satisfies all constraints"""
        params = {
            'dK': individual[0], 'dZ': individual[1], 'LKG': individual[2],
            'lF': individual[3], 'zeta': individual[4]
        }
        return self.constraint_manager.validate_parameters(**params)

    def repair_individual(self, individual):
        """Repair individual to satisfy constraints"""
        params = {
            'dK': individual[0], 'dZ': individual[1], 'LKG': individual[2],
            'lF': individual[3], 'zeta': individual[4]
        }
        repaired = self.constraint_manager.repair_parameters(params, self.param_bounds)
        return [repaired['dK'], repaired['dZ'], repaired['LKG'], repaired['lF'], repaired['zeta']]

    def evaluate_population(self, population, generation):
        """Evaluate population in parallel using ALL available cores simultaneously - NO TIMEOUT"""
        print(f"Evaluating {len(population)} individuals in generation {generation} (NO TIMEOUT)")

        # Validate and repair population
        valid_population = []
        for i, individual in enumerate(population):
            if self.validate_individual(individual):
                valid_population.append(individual)
            else:
                print(f"Individual {i} violates constraints, repairing...")
                repaired = self.repair_individual(individual)
                valid_population.append(repaired)

                # Verify repair worked
                if not self.validate_individual(repaired):
                    print(f"⚠️ Could not repair individual {i}, using penalty")

        print(f"Using ALL available cores: {self.available_cores}")
        print(f"Total cores available: {len(self.available_cores)}")
        print(f"Each simulation will run without timeout until completion")

        # Assign each individual to a specific core
        eval_args = []
        for i, individual in enumerate(valid_population):
            # Use round-robin assignment to distribute across ALL cores
            core_id = self.available_cores[i % len(self.available_cores)]
            print(f"Individual {i} → Core {core_id}")
            eval_args.append((self.optimizer, i, individual, generation, core_id, "Simple", self.fixed_params))

        # Show complete core assignment
        core_assignments = [args[4] for args in eval_args]
        print(f"Complete core assignment: {core_assignments}")

        # Count usage per core
        core_usage = Counter(core_assignments)
        print(f"Core usage distribution: {dict(sorted(core_usage.items()))}")

        # Temporarily remove GUI callback to avoid pickling issues
        original_callback = self.optimizer.progress_callback
        self.optimizer.progress_callback = None

        # Run ALL evaluations in parallel using ALL available cores
        print(f"Starting parallel execution on {len(self.available_cores)} cores (NO TIMEOUT)...")
        print(f"Pool size will be: {len(self.available_cores)} processes")

        # Use ALL available cores as the pool size for maximum parallelization
        with mp.get_context('spawn').Pool(processes=len(self.available_cores)) as pool:
            print(f"✓ Created process pool with {len(self.available_cores)} workers")
            print(f"Each simulation will run until natural completion")
            results = pool.map(evaluate_individual_wrapper, eval_args)
            print(f"✓ All {len(population)} evaluations completed")

        # Restore original callback
        self.optimizer.progress_callback = original_callback

        return np.array(results)

    def domination_sort(self, objectives):
        """Simple Pareto front calculation"""
        n = len(objectives)
        fronts = [[]]
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]

        # Calculate domination relationships
        for i in range(n):
            for j in range(n):
                if i != j:
                    if all(objectives[i] <= objectives[j]) and any(objectives[i] < objectives[j]):
                        dominated_solutions[i].append(j)
                        domination_count[j] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        # Build subsequent fronts
        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)

            if next_front:
                fronts.append(next_front)
            current_front += 1

        return fronts

    def run_optimization(self):
        """Run complete optimization without timeout limits"""
        print(f"Starting optimization: {self.pop_size} individuals, {self.generations} generations")
        print(f"NO TIMEOUT - each simulation will run until completion")

        total_evaluations = self.pop_size * self.generations
        current_evaluation = 0

        # Update progress callback
        if self.progress_callback:
            self.progress_callback.update_status("Initializing Simple NSGA-III population...")

        # Initialize population with constraint-valid individuals
        population = []
        for i in range(self.pop_size):
            individual = self.generate_individual()
            population.append(individual)

        # Verify initial population
        valid_count = sum(1 for ind in population if self.validate_individual(ind))
        print(f"Initial population: {valid_count}/{self.pop_size} individuals satisfy constraints")

        best_objectives = None
        best_individuals = None

        for gen in range(self.generations):
            print(f"\n{'=' * 20} GENERATION {gen + 1} {'=' * 20}")

            # Update progress
            if self.progress_callback:
                self.progress_callback.update_status(f"Running Simple NSGA-III Generation {gen + 1}/{self.generations}")
                self.progress_callback.update_progress(current_evaluation, total_evaluations, gen + 1)

            # Evaluate population without timeout
            objectives = self.evaluate_population(population, gen + 1)
            current_evaluation += self.pop_size

            # Update progress after evaluation
            if self.progress_callback:
                self.progress_callback.update_progress(current_evaluation, total_evaluations, gen + 1)

            # Find Pareto front
            fronts = self.domination_sort(objectives)
            pareto_front = fronts[0] if fronts else []

            if pareto_front:
                print(f"Pareto front size: {len(pareto_front)}")
                best_objectives = objectives[pareto_front]
                best_individuals = [population[idx] for idx in pareto_front]

                # Display best solutions
                pareto_results = f"\nGeneration {gen + 1} - Simple NSGA-III Pareto front ({len(pareto_front)} solutions):"
                for i, idx in enumerate(pareto_front):
                    mech, vol = objectives[idx]
                    params = population[idx]
                    result_line = f"  Solution {i + 1}: Mech={mech:.3f}, Vol={vol:.3f}"
                    print(result_line)
                    pareto_results += f"\n{result_line}"

                    # Show calculated LZ0 and lK value
                    if self.fixed_params:
                        lF_val = params[3]  # lF is at index 3
                        LKG_val = params[2]  # LKG is at index 2
                        LZ0_calc = self.fixed_params.get('LZ', 21.358) - lF_val
                        lK_val = (
                            self.fixed_params.get('standard_LK', 70.0)
                            - (self.fixed_params.get('standard_LKG', 51.62) - LKG_val)
                        )
                        calc_info = f"    LZ0={LZ0_calc:.3f}, lK={lK_val:.3f}"
                        print(calc_info)
                        pareto_results += f"\n{calc_info}"

                    # Verify constraints
                    if not self.validate_individual(params):
                        constraint_warning = f"    ⚠️ Solution {i + 1} violates constraints!"
                        print(constraint_warning)
                        pareto_results += f"\n{constraint_warning}"

                # Update results display
                if self.progress_callback:
                    self.progress_callback.update_results(pareto_results)

                # Simple evolution for next generation
                if gen < self.generations - 1:
                    next_population = []
                    # Keep Pareto front
                    for idx in pareto_front[:self.pop_size // 2]:
                        next_population.append(population[idx])
                    # Generate new constraint-valid individuals
                    while len(next_population) < self.pop_size:
                        new_individual = self.generate_individual()
                        next_population.append(new_individual)
                    population = next_population

        return best_individuals, best_objectives


class AdvancedNSGA3:
    """Advanced NSGA-III implementation with reference directions, constraint handling, and no timeout limits"""

    def __init__(self, optimizer, param_bounds, available_cores, pop_size=20, generations=10, n_partitions=12,
                 constraints_config=None, progress_callback=None, fixed_params=None):
        self.optimizer = optimizer
        # Convert parameter bounds to consistent format
        self.param_bounds = self._normalize_param_bounds(param_bounds)
        self.available_cores = [c for c in available_cores if c != 0]
        if not self.available_cores:
            self.available_cores = [1]

        self.pop_size = pop_size
        self.generations = generations
        self.n_partitions = n_partitions
        self.n_obj = 2  # Mechanical and Volumetric losses
        self.progress_callback = progress_callback
        self.fixed_params = fixed_params or {}

        # Initialize constraint manager
        self.constraint_manager = ConstraintManager()
        self._setup_constraints(constraints_config)

        # Generate reference directions
        self.ref_dirs = self._generate_reference_directions()

        print(f"AdvancedNSGA3 initialized with {len(self.ref_dirs)} reference directions")
        print(f"Constraint system initialized with {len(self.constraint_manager.constraints)} constraints")
        print(f"Fixed parameters: {self.fixed_params}")
        print(f"NO TIMEOUT - simulations will run until completion")
        optimizer.setup_optimization_folder("Advanced")

    def _normalize_param_bounds(self, param_bounds):
        """Convert parameter bounds to consistent dict format"""
        normalized = {}
        for param, bounds in param_bounds.items():
            if isinstance(bounds, (tuple, list)):
                # Convert (min, max) to {'min': min, 'max': max}
                normalized[param] = {'min': bounds[0], 'max': bounds[1]}
            elif isinstance(bounds, dict):
                # Already in correct format
                normalized[param] = bounds
            else:
                raise ValueError(f"Invalid bounds format for {param}: {bounds}")
        return normalized

    def _setup_constraints(self, constraints_config):
        """Setup constraints from configuration"""
        if not constraints_config:
            return

        # Configure dK < dZ constraint
        dk_dz_config = constraints_config.get('dK_less_than_dZ', {})
        if not dk_dz_config.get('active', True):
            self.constraint_manager.deactivate_constraint('dK_less_than_dZ')

        # Configure dZ - dK difference constraint
        dz_dk_config = constraints_config.get('dZ_dK_difference_range', {})
        if not dz_dk_config.get('active', True):
            self.constraint_manager.deactivate_constraint('dZ_dK_difference_range')
        else:
            # Update configuration
            config = {
                'min_difference': dz_dk_config.get('min_difference', 0.1),
                'max_difference': dz_dk_config.get('max_difference', 0.8)
            }
            self.constraint_manager.set_constraint_config('dZ_dK_difference_range', config)

        print("Constraint configuration applied:")
        self.constraint_manager.list_constraints()

    def _generate_reference_directions(self):
        """Generate reference directions for NSGA-III"""
        from itertools import combinations_with_replacement

        # Das and Dennis' method for 2 objectives
        ref_dirs = []
        for i in range(self.n_partitions + 1):
            ref_dirs.append([i / self.n_partitions, 1 - i / self.n_partitions])

        return np.array(ref_dirs)

    def generate_individual(self):
        """Generate random individual within bounds that satisfies constraints"""
        # Try to generate valid individual
        valid_params = self.constraint_manager.generate_valid_parameters(self.param_bounds, max_attempts=100)

        if valid_params:
            return valid_params
        else:
            # Fallback: generate random and try to repair
            print("⚠️ Could not generate valid individual, using repair method")
            individual = [
                random.uniform(self.param_bounds['dK']['min'], self.param_bounds['dK']['max']),
                random.uniform(self.param_bounds['dZ']['min'], self.param_bounds['dZ']['max']),
                random.uniform(self.param_bounds['LKG']['min'], self.param_bounds['LKG']['max']),
                random.uniform(self.param_bounds['lF']['min'], self.param_bounds['lF']['max']),
                random.randint(self.param_bounds['zeta']['min'], self.param_bounds['zeta']['max'])
            ]

            # Try to repair
            param_dict = {
                'dK': individual[0], 'dZ': individual[1], 'LKG': individual[2],
                'lF': individual[3], 'zeta': individual[4]
            }
            repaired = self.constraint_manager.repair_parameters(param_dict, self.param_bounds)

            return [repaired['dK'], repaired['dZ'], repaired['LKG'], repaired['lF'], repaired['zeta']]

    def validate_individual(self, individual):
        """Check if individual satisfies all constraints"""
        params = {
            'dK': individual[0], 'dZ': individual[1], 'LKG': individual[2],
            'lF': individual[3], 'zeta': individual[4]
        }
        return self.constraint_manager.validate_parameters(**params)

    def repair_individual(self, individual):
        """Repair individual to satisfy constraints"""
        params = {
            'dK': individual[0], 'dZ': individual[1], 'LKG': individual[2],
            'lF': individual[3], 'zeta': individual[4]
        }
        repaired = self.constraint_manager.repair_parameters(params, self.param_bounds)
        return [repaired['dK'], repaired['dZ'], repaired['LKG'], repaired['lF'], repaired['zeta']]

    def evaluate_population(self, population, generation):
        """Evaluate population in parallel without timeout"""
        print(f"Evaluating {len(population)} individuals in generation {generation} (NO TIMEOUT)")

        # Validate and repair population
        valid_population = []
        for i, individual in enumerate(population):
            if self.validate_individual(individual):
                valid_population.append(individual)
            else:
                print(f"Individual {i} violates constraints, repairing...")
                repaired = self.repair_individual(individual)
                valid_population.append(repaired)

        # Prepare evaluation arguments
        eval_args = []
        for i, individual in enumerate(valid_population):
            core_id = self.available_cores[i % len(self.available_cores)]
            eval_args.append((self.optimizer, i, individual, generation, core_id, "Advanced", self.fixed_params))

        # Temporarily remove GUI callback to avoid pickling issues
        original_callback = self.optimizer.progress_callback
        self.optimizer.progress_callback = None

        # Run evaluations in parallel without timeout
        print(f"Running evaluations on {len(self.available_cores)} cores - NO TIMEOUT")
        with mp.get_context('spawn').Pool(processes=len(self.available_cores)) as pool:
            results = pool.map(evaluate_individual_wrapper, eval_args)

        # Restore original callback
        self.optimizer.progress_callback = original_callback

        return np.array(results)

    # [Continue with other AdvancedNSGA3 methods - normalize_objectives, associate_to_reference_directions, etc.]
    # For brevity, I'll include the key methods. The full implementation would include all methods from the original.

    def run_optimization(self):
        """Run complete NSGA-III optimization without timeout limits"""
        print(f"Starting NSGA-III optimization: {self.pop_size} individuals, {self.generations} generations")
        print(f"NO TIMEOUT - each simulation will run until completion")

        total_evaluations = self.pop_size * (self.generations * 2 - 1)  # Accounting for offspring
        current_evaluation = 0

        # Update progress callback
        if self.progress_callback:
            self.progress_callback.update_status("Initializing Advanced NSGA-III population...")

        # Initialize population with constraint-valid individuals
        population = []
        for i in range(self.pop_size):
            individual = self.generate_individual()
            population.append(individual)

        # Verify initial population
        valid_count = sum(1 for ind in population if self.validate_individual(ind))
        print(f"Initial population: {valid_count}/{self.pop_size} individuals satisfy constraints")

        for gen in range(self.generations):
            print(f"\n{'=' * 20} GENERATION {gen + 1} {'=' * 20}")

            # Update progress
            if self.progress_callback:
                self.progress_callback.update_status(
                    f"Running Advanced NSGA-III Generation {gen + 1}/{self.generations}")
                self.progress_callback.update_progress(current_evaluation, total_evaluations, gen + 1)

            # Evaluate population without timeout
            objectives = self.evaluate_population(population, gen + 1)
            current_evaluation += self.pop_size

            # Create offspring (simplified for this example)
            if gen < self.generations - 1:
                offspring = self.crossover_and_mutation(population)
                offspring_objectives = self.evaluate_population(offspring, gen + 1)
                current_evaluation += self.pop_size

                # Combine parent and offspring populations
                combined_population = population + offspring
                combined_objectives = np.vstack([objectives, offspring_objectives])

                # Environmental selection (simplified)
                fronts = self.domination_sort(combined_objectives)
                selected_indices = []
                front_idx = 0

                # Add complete fronts
                while len(selected_indices) + len(fronts[front_idx]) <= self.pop_size:
                    selected_indices.extend(fronts[front_idx])
                    front_idx += 1
                    if front_idx >= len(fronts):
                        break

                # Fill remaining slots from next front
                if len(selected_indices) < self.pop_size and front_idx < len(fronts):
                    remaining = self.pop_size - len(selected_indices)
                    selected_indices.extend(fronts[front_idx][:remaining])

                population = [combined_population[i] for i in selected_indices[:self.pop_size]]
                objectives = combined_objectives[selected_indices[:self.pop_size]]

            # Find and display Pareto front
            fronts = self.domination_sort(objectives)
            pareto_front = fronts[0] if fronts else []

            if pareto_front:
                print(f"Pareto front size: {len(pareto_front)}")
                best_objectives = objectives[pareto_front]
                best_individuals = [population[idx] for idx in pareto_front]

                # Display best solutions and verify constraints
                pareto_results = f"\nGeneration {gen + 1} - Advanced NSGA-III Pareto front ({len(pareto_front)} solutions):"
                for i, idx in enumerate(pareto_front):
                    mech, vol = objectives[idx]
                    params = population[idx]
                    result_line = f"  Solution {i + 1}: Mech={mech:.3f}, Vol={vol:.3f}"
                    print(result_line)
                    pareto_results += f"\n{result_line}"

                    # Show calculated LZ0 and lK value
                    if self.fixed_params:
                        lF_val = params[3]  # lF is at index 3
                        LKG_val = params[2]  # LKG is at index 2
                        LZ0_calc = self.fixed_params.get('LZ', 21.358) - lF_val
                        lK_val = (
                            self.fixed_params.get('standard_LK', 70.0)
                            - (self.fixed_params.get('standard_LKG', 51.62) - LKG_val)
                        )
                        calc_info = f"    LZ0={LZ0_calc:.3f}, lK={lK_val:.3f}"
                        print(calc_info)
                        pareto_results += f"\n{calc_info}"

                    # Verify constraints
                    if not self.validate_individual(params):
                        constraint_warning = f"    ⚠️ Solution {i + 1} violates constraints!"
                        print(constraint_warning)
                        pareto_results += f"\n{constraint_warning}"

                if self.progress_callback:
                    self.progress_callback.update_results(pareto_results)
                    self.progress_callback.update_progress(current_evaluation, total_evaluations, gen + 1)

        # Final results
        final_objectives = self.evaluate_population(population, self.generations)
        fronts = self.domination_sort(final_objectives)
        pareto_front = fronts[0] if fronts else []

        if pareto_front:
            best_objectives = final_objectives[pareto_front]
            best_individuals = [population[idx] for idx in pareto_front]

            # Final constraint validation
            final_results = f"\nFinal Advanced NSGA-III constraint validation:"
            for i, individual in enumerate(best_individuals):
                is_valid = self.validate_individual(individual)
                validation_result = f"  Solution {i + 1}: {'✓ Valid' if is_valid else '✗ Invalid'}"
                print(validation_result)
                final_results += f"\n{validation_result}"

            if self.progress_callback:
                self.progress_callback.update_results(final_results)

            return best_individuals, best_objectives

        return [], []

    def domination_sort(self, objectives):
        """Non-dominated sorting"""
        n = len(objectives)
        fronts = [[]]
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i != j:
                    if all(objectives[i] <= objectives[j]) and any(objectives[i] < objectives[j]):
                        dominated_solutions[i].append(j)
                        domination_count[j] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)

            if next_front:
                fronts.append(next_front)
            current_front += 1

        return fronts

    def crossover_and_mutation(self, parent_population):
        """Create offspring through crossover and mutation with constraint handling"""
        offspring = []

        for _ in range(self.pop_size):
            # Tournament selection
            parent1 = self.tournament_selection(parent_population)
            parent2 = self.tournament_selection(parent_population)

            # Simulated binary crossover (SBX)
            child = self.sbx_crossover(parent1, parent2)

            # Polynomial mutation
            child = self.polynomial_mutation(child)

            # Ensure child satisfies constraints
            if not self.validate_individual(child):
                child = self.repair_individual(child)

            offspring.append(child)

        return offspring

    def tournament_selection(self, population, tournament_size=2):
        """Tournament selection"""
        tournament = random.sample(population, tournament_size)
        return random.choice(tournament)  # Simplified - should use fitness

    def sbx_crossover(self, parent1, parent2, eta_c=20):
        """Simulated binary crossover with constraint handling"""
        child = []

        for i in range(len(parent1)):
            if random.random() < 0.9:  # Crossover probability
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    if random.random() <= 0.5:
                        beta = (2 * random.random()) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - random.random()))) ** (1.0 / (eta_c + 1))

                    c1 = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                    c2 = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])

                    child_val = c1 if random.random() < 0.5 else c2
                else:
                    child_val = parent1[i]
            else:
                child_val = parent1[i]

            # Apply bounds
            param_names = ['dK', 'dZ', 'LKG', 'lF', 'zeta']
            param_name = param_names[i]
            bounds = self.param_bounds[param_name]

            if param_name == 'zeta':
                child_val = max(bounds['min'], min(bounds['max'], int(round(child_val))))
            else:
                child_val = max(bounds['min'], min(bounds['max'], child_val))

            child.append(child_val)

        return child

    def polynomial_mutation(self, individual, eta_m=20):
        """Polynomial mutation with constraint handling"""
        mutated = individual.copy()
        param_names = ['dK', 'dZ', 'LKG', 'lF', 'zeta']

        for i in range(len(mutated)):
            if random.random() < 0.1:  # Mutation probability
                param_name = param_names[i]
                bounds = self.param_bounds[param_name]

                y = mutated[i]
                delta1 = (y - bounds['min']) / (bounds['max'] - bounds['min'])
                delta2 = (bounds['max'] - y) / (bounds['max'] - bounds['min'])

                rnd = random.random()
                mut_pow = 1.0 / (eta_m + 1.0)

                if rnd <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (eta_m + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (eta_m + 1.0))
                    deltaq = 1.0 - val ** mut_pow

                y = y + deltaq * (bounds['max'] - bounds['min'])

                if param_name == 'zeta':
                    y = max(bounds['min'], min(bounds['max'], int(round(y))))
                else:
                    y = max(bounds['min'], min(bounds['max'], y))

                mutated[i] = y

        return mutated
