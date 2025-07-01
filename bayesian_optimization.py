"""
Bayesian Optimization Algorithm - Complete Fixed Version
======================================================

Fixed all issues:
1. Proper folder management (single folder)
2. Correct worker count (batch size = worker count)
3. Enhanced error handling and debugging
4. Proper failure detection and reporting
5. Robust constraint handling
6. Better logging for troubleshooting
"""

import numpy as np
import random
import multiprocessing as mp
import psutil
import traceback
import os
from pathlib import Path
from collections import Counter
from cpu_affinity import get_available_cores
from constraints import ConstraintManager
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def evaluate_individual_wrapper_bo(args):
    """Wrapper for multiprocessing evaluation - Enhanced with proper error handling"""
    base_folder, individual_id, params, iteration, core_id, fixed_params, optimization_folder = args

    print(f"BO Worker {individual_id} starting on core {core_id}")

    try:
        # Set current worker process to exclude core 0
        try:
            current_process = psutil.Process()
            total_cores = psutil.cpu_count()
            all_cores_except_0 = list(range(1, total_cores))

            current_process.cpu_affinity(all_cores_except_0)
            final_affinity = current_process.cpu_affinity()
            print(f"BO Worker {individual_id} process affinity: {final_affinity}")

            if 0 in final_affinity:
                print(f"WARNING: BO Worker {individual_id} still has core 0, correcting...")
                corrected = [c for c in final_affinity if c != 0]
                current_process.cpu_affinity(corrected)
                final_affinity = current_process.cpu_affinity()
                print(f"BO Worker {individual_id} corrected affinity: {final_affinity}")

        except Exception as e:
            print(f"Warning: Could not set BO worker {individual_id} process affinity: {e}")

        # Verify base folder exists
        base_path = Path(base_folder)
        if not base_path.exists():
            raise FileNotFoundError(f"Base folder does not exist: {base_folder}")

        # Verify required files exist
        exe_path = base_path / 'fsti_gap.exe'
        input_path = base_path / 'input'

        if not exe_path.exists():
            raise FileNotFoundError(f"fsti_gap.exe not found: {exe_path}")
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_path}")

        print(f"BO Worker {individual_id}: Base folder verified: {base_folder}")
        print(f"BO Worker {individual_id}: fsti_gap.exe found: {exe_path}")
        print(f"BO Worker {individual_id}: Input folder found: {input_path}")

        # Create worker optimizer using the EXISTING optimization folder
        from piston_optimizer import PistonOptimizer
        worker_optimizer = PistonOptimizer(base_folder, progress_callback=None)

        # Use existing folder instead of creating new one
        worker_optimizer.algorithm_folder = Path(optimization_folder)
        worker_optimizer.nsga3_folder = Path(optimization_folder)  # For legacy compatibility

        print(f"BO Worker {individual_id} using optimization folder: {optimization_folder}")

        # Verify optimization folder exists
        if not worker_optimizer.algorithm_folder.exists():
            raise FileNotFoundError(f"Optimization folder does not exist: {optimization_folder}")

        # Run evaluation with detailed parameter logging
        dK, dZ, LKG, lF, zeta = params
        print(f"BO Worker {individual_id} evaluating: dK={dK:.6f}, dZ={dZ:.6f}, LKG={LKG:.6f}, lF={lF:.6f}, zeta={int(zeta)}")

        # Calculate and display LZ0 and LKG values
        if fixed_params:
            LZ0_calc = fixed_params.get('LZ', 21.358) - lF
            lK_val = (
                fixed_params.get('standard_LK', 70.0)
                + (fixed_params.get('standard_LKG', 51.62) - LKG)
            )
            print(f"BO Worker {individual_id} calculations: LZ0={LZ0_calc:.6f}, lK={lK_val:.6f}")

        result = worker_optimizer.evaluate_individual(
            individual_id, dK, dZ, LKG, lF, zeta, iteration,
            algorithm_type="Bayesian", fixed_params=fixed_params
        )

        mech_loss, vol_loss = result
        print(f"BO Worker {individual_id} completed: Mech={mech_loss:.6f}, Vol={vol_loss:.6f}")

        # Check if result indicates failure
        if mech_loss >= 1e5 or vol_loss >= 1e5:
            print(f"BO Worker {individual_id} FAILED: Penalty values detected")
            return 1e6, 1e6  # Return consistent penalty values

        return mech_loss, vol_loss

    except Exception as e:
        print(f"BO Worker {individual_id} ERROR: {str(e)}")
        print(f"BO Worker {individual_id} ERROR traceback:")
        traceback.print_exc()
        return 1e6, 1e6  # Return penalty values on error


class BayesianOptimization:
    """Bayesian Optimization with enhanced error handling and debugging"""

    def __init__(self, optimizer, param_bounds, available_cores, n_initial=10, n_iterations=20,
                 acquisition='ei', constraints_config=None, alpha=1e-6, progress_callback=None,
                 fixed_params=None, batch_size=None):
        self.optimizer = optimizer
        # Convert parameter bounds to consistent format
        self.param_bounds = self._normalize_param_bounds(param_bounds)
        self.available_cores = [c for c in available_cores if c != 0]
        if not self.available_cores:
            self.available_cores = [1]

        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.acquisition = acquisition.lower()
        self.alpha = alpha  # Noise parameter for GP
        self.progress_callback = progress_callback
        self.fixed_params = fixed_params or {}

        # Set batch size (number of individuals to evaluate in parallel per iteration)
        if batch_size is None:
            # Default: use min of available cores or reasonable batch size
            self.batch_size = min(len(self.available_cores), 8)
        else:
            self.batch_size = max(1, min(batch_size, len(self.available_cores)))

        # Parameter info
        self.param_names = ['dK', 'dZ', 'LKG', 'lF', 'zeta']
        self.n_params = len(self.param_names)

        # Create bounds arrays for optimization
        self.bounds_array = np.array([
            [self.param_bounds[param]['min'], self.param_bounds[param]['max']]
            for param in self.param_names
        ])

        # Initialize constraint manager
        self.constraint_manager = ConstraintManager()
        self._setup_constraints(constraints_config)

        # Data storage
        self.X_observed = []  # Parameter vectors
        self.Y_observed = []  # Objective values [mechanical_loss, volumetric_loss]
        self.pareto_front = []  # Current Pareto optimal solutions
        self.pareto_objectives = []  # Corresponding objectives

        # Track evaluation failures
        self.failed_evaluations = 0
        self.total_evaluations = 0

        # Gaussian Process models (one for each objective)
        self.gp_mechanical = None
        self.gp_volumetric = None

        print(f"BayesianOptimization initialized (BATCH PARALLEL MODE)")
        print(f"Available cores: {self.available_cores} (total: {len(self.available_cores)})")
        print(f"Requested batch size: {batch_size}")
        print(f"Actual batch size: {self.batch_size} individuals per iteration")
        print(f"Max parallel workers: {min(self.batch_size, len(self.available_cores))}")
        print(f"Initial samples: {self.n_initial}, Iterations: {self.n_iterations}")
        print(f"Acquisition function: {self.acquisition}")
        print(f"Constraint system initialized with {len(self.constraint_manager.constraints)} constraints")
        print(f"Fixed parameters: {self.fixed_params}")
        print(f"NO TIMEOUT - each simulation will run until completion")

        # IMPORTANT: Setup optimization folder ONCE in main thread
        optimizer.setup_optimization_folder("Bayesian")
        print(f"✓ Single optimization folder created: {optimizer.algorithm_folder}")

        # Verify setup
        self._verify_setup()

    def _verify_setup(self):
        """Verify that all required files and folders exist"""
        print("Verifying optimization setup...")

        # Check base folder
        base_path = Path(self.optimizer.base_folder)
        if not base_path.exists():
            raise FileNotFoundError(f"Base folder does not exist: {base_path}")

        # Check required files
        exe_path = base_path / 'fsti_gap.exe'
        input_path = base_path / 'input'

        if not exe_path.exists():
            raise FileNotFoundError(f"fsti_gap.exe not found: {exe_path}")
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_path}")

        # Check optimization folder
        if not self.optimizer.algorithm_folder.exists():
            raise FileNotFoundError(f"Optimization folder not created: {self.optimizer.algorithm_folder}")

        print(f"✓ Base folder verified: {base_path}")
        print(f"✓ fsti_gap.exe found: {exe_path}")
        print(f"✓ Input folder found: {input_path}")
        print(f"✓ Optimization folder ready: {self.optimizer.algorithm_folder}")

    def _normalize_param_bounds(self, param_bounds):
        """Convert parameter bounds to consistent dict format"""
        normalized = {}
        for param, bounds in param_bounds.items():
            if isinstance(bounds, (tuple, list)):
                normalized[param] = {'min': bounds[0], 'max': bounds[1]}
            elif isinstance(bounds, dict):
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
            config = {
                'min_difference': dz_dk_config.get('min_difference', 0.1),
                'max_difference': dz_dk_config.get('max_difference', 0.8)
            }
            self.constraint_manager.set_constraint_config('dZ_dK_difference_range', config)

        print("BO Constraint configuration applied:")
        self.constraint_manager.list_constraints()

    def generate_individual(self):
        """Generate constraint-valid individual"""
        valid_params = self.constraint_manager.generate_valid_parameters(self.param_bounds, max_attempts=100)

        if valid_params:
            return np.array(valid_params)
        else:
            # Fallback: generate random and repair
            print("⚠️ BO: Could not generate valid individual, using repair method")

            individual = np.array([
                random.uniform(self.param_bounds['dK']['min'], self.param_bounds['dK']['max']),
                random.uniform(self.param_bounds['dZ']['min'], self.param_bounds['dZ']['max']),
                random.uniform(self.param_bounds['LKG']['min'], self.param_bounds['LKG']['max']),
                random.uniform(self.param_bounds['lF']['min'], self.param_bounds['lF']['max']),
                random.randint(self.param_bounds['zeta']['min'], self.param_bounds['zeta']['max'])
            ])


            param_dict = {self.param_names[i]: individual[i] for i in range(len(self.param_names))}
            repaired = self.constraint_manager.repair_parameters(param_dict, self.param_bounds)

            return np.array([repaired[param] for param in self.param_names])

    def validate_individual(self, individual):
        """Check if individual satisfies all constraints"""
        params = {self.param_names[i]: individual[i] for i in range(len(self.param_names))}
        return self.constraint_manager.validate_parameters(**params)

    def repair_individual(self, individual):
        """Repair individual to satisfy constraints"""
        params = {self.param_names[i]: individual[i] for i in range(len(self.param_names))}
        repaired = self.constraint_manager.repair_parameters(params, self.param_bounds)
        return np.array([repaired[param] for param in self.param_names])

    def evaluate_batch_parallel(self, X_batch, iteration):
        """Evaluate batch of individuals in parallel - Enhanced with failure detection"""
        print(f"\n{'='*60}")
        print(f"BO: Evaluating BATCH of {len(X_batch)} individuals in parallel (iteration {iteration})")
        print(f"BO: Available cores: {self.available_cores}")
        print(f"BO: Will use UP TO {min(len(X_batch), len(self.available_cores))} worker processes")
        print(f"{'='*60}")

        # Update progress callback in main thread
        if self.progress_callback:
            for i, individual in enumerate(X_batch):
                self.progress_callback.update_current_evaluation(i, individual.tolist())

        # Validate and repair batch
        valid_batch = []
        for i, individual in enumerate(X_batch):
            if self.validate_individual(individual):
                valid_batch.append(individual)
                print(f"BO: Individual {i} passed constraint validation")
            else:
                print(f"BO: Individual {i} violates constraints, repairing...")
                repaired = self.repair_individual(individual)
                valid_batch.append(repaired)

                # Verify repair worked
                if self.validate_individual(repaired):
                    print(f"BO: Individual {i} successfully repaired")
                else:
                    print(f"⚠️ BO: Individual {i} repair failed - using anyway")

        # Display batch details
        print(f"\nBO: Batch Details:")
        for i, individual in enumerate(valid_batch):
            dK, dZ, LKG, lF, zeta = individual
            print(f"  Individual {i}: dK={dK:.4f}, dZ={dZ:.4f}, LKG={LKG:.2f}, lF={lF:.2f}, zeta={int(zeta)}")

            # Show calculated values
            if self.fixed_params:
                LZ0_calc = self.fixed_params.get('LZ', 21.358) - lF
                lK_val = (
                    self.fixed_params.get('standard_LK', 70.0)
                    + (self.fixed_params.get('standard_LKG', 51.62) - LKG)
                )
                print(f"    Calculated: LZ0={LZ0_calc:.4f}, lK={lK_val:.4f}")

        # Prepare evaluation arguments
        eval_args = []
        for i, individual in enumerate(valid_batch):
            core_id = self.available_cores[i % len(self.available_cores)]
            eval_args.append((
                str(self.optimizer.base_folder),
                i,
                individual.tolist(),
                iteration,
                core_id,
                self.fixed_params,
                str(self.optimizer.algorithm_folder)
            ))

        print(f"\nBO: Process assignments:")
        for i, (_, ind_id, params, _, core_id, _, folder) in enumerate(eval_args):
            print(f"  Individual {ind_id} → Core {core_id}")

        # Use optimal number of workers
        max_workers = min(len(valid_batch), len(self.available_cores))
        print(f"\nBO: Creating process pool with {max_workers} workers")
        print(f"    Batch size: {len(valid_batch)}")
        print(f"    Available cores: {len(self.available_cores)}")

        # Run batch evaluation in parallel
        try:
            with mp.get_context('spawn').Pool(processes=max_workers) as pool:
                print(f"BO: Starting parallel evaluation...")
                results = pool.map(evaluate_individual_wrapper_bo, eval_args)
                print(f"BO: Parallel evaluation completed")
        except Exception as e:
            print(f"BO: ERROR in parallel evaluation: {e}")
            traceback.print_exc()
            # Return penalty values for all individuals
            results = [(1e6, 1e6) for _ in valid_batch]

        # Analyze results
        successful_results = 0
        failed_results = 0

        print(f"\nBO: Batch Results Analysis:")
        for i, result in enumerate(results):
            mech_loss, vol_loss = result
            if mech_loss >= 1e5 or vol_loss >= 1e5:
                failed_results += 1
                print(f"  Individual {i}: FAILED (Mech={mech_loss:.0f}, Vol={vol_loss:.0f})")
            else:
                successful_results += 1
                print(f"  Individual {i}: SUCCESS (Mech={mech_loss:.6f}, Vol={vol_loss:.6f})")

        self.total_evaluations += len(results)
        self.failed_evaluations += failed_results

        success_rate = (successful_results / len(results)) * 100 if results else 0
        overall_success_rate = ((self.total_evaluations - self.failed_evaluations) / self.total_evaluations) * 100 if self.total_evaluations > 0 else 0

        print(f"\nBO: Batch Success Rate: {success_rate:.1f}% ({successful_results}/{len(results)})")
        print(f"BO: Overall Success Rate: {overall_success_rate:.1f}% ({self.total_evaluations - self.failed_evaluations}/{self.total_evaluations})")

        # Update progress in main thread after completion
        if self.progress_callback:
            for i, result in enumerate(results):
                individual = valid_batch[i]
                mech_loss, vol_loss = result
                if mech_loss < 1e5 and vol_loss < 1e5:
                    self.progress_callback.update_results(
                        f"BO Individual {i}: SUCCESS - Mech={mech_loss:.3f}, Vol={vol_loss:.3f}")
                else:
                    self.progress_callback.update_results(f"BO Individual {i}: FAILED")

        print(f"✓ BO: Batch evaluation completed using {max_workers} workers")
        print(f"{'='*60}\n")

        return np.array(results)

    def fit_gaussian_processes(self):
        """Fit Gaussian Process models to observed data - only use successful evaluations"""
        if len(self.Y_observed) < 2:
            print("BO: Not enough data to fit Gaussian Processes")
            return

        X = np.array(self.X_observed)
        Y = np.array(self.Y_observed)

        # Filter out failed evaluations (penalty values)
        valid_mask = (Y[:, 0] < 1e5) & (Y[:, 1] < 1e5)

        if np.sum(valid_mask) < 2:
            print("BO: Not enough successful evaluations to fit Gaussian Processes")
            return

        X_valid = X[valid_mask]
        Y_valid = Y[valid_mask]

        print(f"BO: Fitting GP with {len(Y_valid)} successful evaluations out of {len(Y)} total")

        # Kernel for GP (RBF + constant + noise)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

        try:
            # Fit GP for mechanical loss
            self.gp_mechanical = GaussianProcessRegressor(
                kernel=kernel, alpha=self.alpha, n_restarts_optimizer=3, random_state=42
            )
            self.gp_mechanical.fit(X_valid, Y_valid[:, 0])

            # Fit GP for volumetric loss
            self.gp_volumetric = GaussianProcessRegressor(
                kernel=kernel, alpha=self.alpha, n_restarts_optimizer=3, random_state=42
            )
            self.gp_volumetric.fit(X_valid, Y_valid[:, 1])

            print("BO: Gaussian Process models fitted successfully")

        except Exception as e:
            print(f"BO: Warning - GP fitting failed: {e}")

    def acquisition_function(self, X, xi=0.01):
        """Compute acquisition function values"""
        if self.gp_mechanical is None or self.gp_volumetric is None:
            # Random acquisition if no GP model yet
            return np.random.rand(X.shape[0])

        X = np.atleast_2d(X)

        try:
            # Get predictions from both GPs
            mu_mech, sigma_mech = self.gp_mechanical.predict(X, return_std=True)
            mu_vol, sigma_vol = self.gp_volumetric.predict(X, return_std=True)

            # Prevent division by zero
            sigma_mech = np.maximum(sigma_mech, 1e-9)
            sigma_vol = np.maximum(sigma_vol, 1e-9)

            if self.acquisition == 'ei':  # Expected Improvement
                # Multi-objective EI: product of EI for each objective
                # Only use successful evaluations for best values
                if len(self.Y_observed) > 0:
                    Y = np.array(self.Y_observed)
                    valid_mask = (Y[:, 0] < 1e5) & (Y[:, 1] < 1e5)

                    if np.any(valid_mask):
                        Y_valid = Y[valid_mask]
                        f_best_mech = np.min(Y_valid[:, 0])
                        f_best_vol = np.min(Y_valid[:, 1])
                    else:
                        # No successful evaluations yet
                        f_best_mech = 0
                        f_best_vol = 0
                else:
                    f_best_mech = 0
                    f_best_vol = 0

                # EI for mechanical loss (minimization)
                imp_mech = f_best_mech - mu_mech - xi
                Z_mech = imp_mech / sigma_mech
                ei_mech = imp_mech * norm.cdf(Z_mech) + sigma_mech * norm.pdf(Z_mech)
                ei_mech[sigma_mech == 0] = 0

                # EI for volumetric loss (minimization)
                imp_vol = f_best_vol - mu_vol - xi
                Z_vol = imp_vol / sigma_vol
                ei_vol = imp_vol * norm.cdf(Z_vol) + sigma_vol * norm.pdf(Z_vol)
                ei_vol[sigma_vol == 0] = 0

                # Combined EI (geometric mean to balance objectives)
                ei_combined = np.sqrt(np.maximum(ei_mech, 0) * np.maximum(ei_vol, 0))
                return ei_combined

            elif self.acquisition == 'pi':  # Probability of Improvement
                if len(self.Y_observed) > 0:
                    Y = np.array(self.Y_observed)
                    valid_mask = (Y[:, 0] < 1e5) & (Y[:, 1] < 1e5)

                    if np.any(valid_mask):
                        Y_valid = Y[valid_mask]
                        f_best_mech = np.min(Y_valid[:, 0])
                        f_best_vol = np.min(Y_valid[:, 1])
                    else:
                        f_best_mech = 0
                        f_best_vol = 0
                else:
                    f_best_mech = 0
                    f_best_vol = 0

                Z_mech = (f_best_mech - mu_mech - xi) / sigma_mech
                Z_vol = (f_best_vol - mu_vol - xi) / sigma_vol

                pi_mech = norm.cdf(Z_mech)
                pi_vol = norm.cdf(Z_vol)

                return np.sqrt(pi_mech * pi_vol)

            elif self.acquisition == 'ucb':  # Upper Confidence Bound
                # For minimization, we want Lower Confidence Bound
                kappa = 2.0  # Exploration parameter
                lcb_mech = mu_mech - kappa * sigma_mech
                lcb_vol = mu_vol - kappa * sigma_vol

                # Convert to maximization problem (negative of sum)
                return -(lcb_mech + lcb_vol)

            else:
                raise ValueError(f"Unknown acquisition function: {self.acquisition}")

        except Exception as e:
            print(f"BO: Warning - Acquisition function computation failed: {e}")
            return np.random.rand(X.shape[0])

    def optimize_acquisition_batch(self, batch_size, n_candidates=1000):
        """Find multiple promising points for batch evaluation"""
        print(f"BO: Finding {batch_size} promising points for batch evaluation")

        # Generate many candidates
        candidates = []
        for _ in range(n_candidates):
            candidate = self.generate_individual()
            candidates.append(candidate)

        if not candidates:
            print("BO: Warning - No valid candidates generated")
            batch = []
            for _ in range(batch_size):
                batch.append(self.generate_individual())
            return np.array(batch)

        X_candidates = np.array(candidates)
        acq_values = self.acquisition_function(X_candidates)

        # Select top batch_size candidates
        if len(acq_values) > 0:
            # Sort by acquisition value (descending)
            sorted_indices = np.argsort(acq_values)[::-1]

            # Select diverse batch (avoid duplicates)
            selected_batch = []
            selected_indices = []

            for idx in sorted_indices:
                candidate = X_candidates[idx]

                # Check if this candidate is too similar to already selected ones
                is_diverse = True
                for selected in selected_batch:
                    # Simple diversity check - ensure parameters aren't too close
                    if np.allclose(candidate, selected, atol=0.01):
                        is_diverse = False
                        break

                if is_diverse:
                    selected_batch.append(candidate)
                    selected_indices.append(idx)

                    if len(selected_batch) >= batch_size:
                        break

            # Fill remaining slots if needed
            while len(selected_batch) < batch_size:
                new_candidate = self.generate_individual()
                selected_batch.append(new_candidate)

            selected_batch = np.array(selected_batch)

            # Show acquisition values for selected batch
            if selected_indices:
                selected_acq_values = [acq_values[i] for i in selected_indices[:len(selected_indices)]]
                print(f"BO: Selected batch with acquisition values: {[f'{v:.6f}' for v in selected_acq_values]}")

            return selected_batch
        else:
            print("BO: Warning - Acquisition optimization failed, using random batch")
            batch = []
            for _ in range(batch_size):
                batch.append(self.generate_individual())
            return np.array(batch)

    def update_pareto_front(self):
        """Update Pareto front from observed data - only include successful evaluations"""
        if len(self.Y_observed) == 0:
            return

        Y = np.array(self.Y_observed)
        X = np.array(self.X_observed)

        # Filter out failed evaluations
        valid_mask = (Y[:, 0] < 1e5) & (Y[:, 1] < 1e5)

        if not np.any(valid_mask):
            print("BO: No successful evaluations for Pareto front")
            self.pareto_front = []
            self.pareto_objectives = []
            return

        Y_valid = Y[valid_mask]
        X_valid = X[valid_mask]

        # Find Pareto optimal points among successful evaluations
        pareto_mask = self._is_pareto_optimal(Y_valid)

        self.pareto_front = X_valid[pareto_mask].tolist()
        self.pareto_objectives = Y_valid[pareto_mask].tolist()

        successful_count = len(X_valid)
        pareto_count = len(self.pareto_front)
        print(f"BO: Pareto front updated - {pareto_count} optimal solutions from {successful_count} successful evaluations")

    def _is_pareto_optimal(self, objectives):
        """Check which points are Pareto optimal (for minimization)"""
        n_points = objectives.shape[0]
        is_optimal = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    # Check if j dominates i (all objectives better or equal, at least one strictly better)
                    if (np.all(objectives[j] <= objectives[i]) and
                            np.any(objectives[j] < objectives[i])):
                        is_optimal[i] = False
                        break

        return is_optimal

    def run_optimization(self):
        """Run Bayesian Optimization with enhanced error handling and reporting"""
        print(f"\n{'='*80}")
        print(f"BAYESIAN OPTIMIZATION STARTING")
        print(f"{'='*80}")
        print(f"BO: Initial samples: {self.n_initial}")
        print(f"BO: BO iterations: {self.n_iterations}")
        print(f"BO: Batch size: {self.batch_size}")
        print(f"BO: Total evaluations: ~{self.n_initial + (self.n_iterations * self.batch_size)}")
        print(f"BO: NO TIMEOUT - each simulation will run until completion")
        print(f"BO: Mode: BATCH PARALLEL")
        print(f"{'='*80}")

        total_evaluations = self.n_initial + (self.n_iterations * self.batch_size)
        current_evaluation = 0

        # Update progress callback
        if self.progress_callback:
            self.progress_callback.update_status("Starting Bayesian Optimization (Batch Parallel)...")

        # Phase 1: Initial sampling batch
        print(f"\n{'-'*60}")
        print(f"PHASE 1: INITIAL SAMPLING BATCH")
        print(f"{'-'*60}")

        if self.progress_callback:
            self.progress_callback.update_status("Running initial sampling batch...")
            self.progress_callback.update_progress(current_evaluation, total_evaluations, 0)

        # Generate initial batch
        print(f"BO: Generating {self.n_initial} initial individuals...")
        initial_X = []
        for i in range(self.n_initial):
            individual = self.generate_individual()
            initial_X.append(individual)
            print(f"  Initial individual {i}: Generated")

        initial_X = np.array(initial_X)

        # Evaluate initial batch in parallel
        print(f"\nBO: Evaluating initial batch of {len(initial_X)} individuals...")
        initial_Y = self.evaluate_batch_parallel(initial_X, iteration=0)
        current_evaluation += self.n_initial

        # Store initial data
        self.X_observed.extend(initial_X.tolist())
        self.Y_observed.extend(initial_Y.tolist())

        # Update Pareto front
        self.update_pareto_front()

        # Analyze initial results
        successful_initial = sum(1 for y in initial_Y if y[0] < 1e5 and y[1] < 1e5)
        initial_success_rate = (successful_initial / len(initial_Y)) * 100

        initial_results = f"Initial batch completed - {len(self.pareto_front)} Pareto optimal solutions found"
        initial_results += f"\nInitial success rate: {initial_success_rate:.1f}% ({successful_initial}/{len(initial_Y)})"

        print(f"\nBO: {initial_results}")
        if self.progress_callback:
            self.progress_callback.update_results(initial_results)
            self.progress_callback.update_progress(current_evaluation, total_evaluations, 0)

        # Check if we have any successful evaluations
        if successful_initial == 0:
            error_msg = "❌ CRITICAL: All initial evaluations failed!"
            error_msg += "\nPossible issues:"
            error_msg += "\n- fsti_gap.exe execution problems"
            error_msg += "\n- Input file configuration errors"
            error_msg += "\n- Path or permission issues"
            error_msg += "\n- Constraint violations causing simulation failures"
            print(f"\n{error_msg}")
            if self.progress_callback:
                self.progress_callback.update_results(error_msg)

            # Still continue to see if later evaluations succeed
            print("\nBO: Continuing optimization to gather more diagnostic information...")

        # Phase 2: BO iterations with batch evaluation
        print(f"\n{'-'*60}")
        print(f"PHASE 2: BAYESIAN OPTIMIZATION ITERATIONS")
        print(f"{'-'*60}")

        for iteration in range(1, self.n_iterations + 1):
            print(f"\n{'='*40}")
            print(f"BO ITERATION {iteration}/{self.n_iterations}")
            print(f"{'='*40}")

            if self.progress_callback:
                self.progress_callback.update_status(f"Running BO Iteration {iteration}/{self.n_iterations} (batch)")
                self.progress_callback.update_progress(current_evaluation, total_evaluations, iteration)

            # Fit Gaussian Process models
            print(f"BO: Fitting Gaussian Process models...")
            self.fit_gaussian_processes()

            # Find batch of promising points
            print(f"BO: Optimizing acquisition function for batch of {self.batch_size}...")
            next_batch_X = self.optimize_acquisition_batch(self.batch_size)

            # Evaluate batch in parallel
            print(f"BO: Evaluating iteration {iteration} batch...")
            next_batch_Y = self.evaluate_batch_parallel(next_batch_X, iteration=iteration)
            current_evaluation += self.batch_size

            # Store new data
            self.X_observed.extend(next_batch_X.tolist())
            self.Y_observed.extend(next_batch_Y.tolist())

            # Update Pareto front
            self.update_pareto_front()

            # Analyze iteration results
            successful_iter = sum(1 for y in next_batch_Y if y[0] < 1e5 and y[1] < 1e5)
            iter_success_rate = (successful_iter / len(next_batch_Y)) * 100

            # Display current best solutions
            if self.pareto_front:
                iteration_results = f"BO Iteration {iteration} completed - Pareto front size: {len(self.pareto_front)}"
                iteration_results += f"\nIteration success rate: {iter_success_rate:.1f}% ({successful_iter}/{len(next_batch_Y)})"
                iteration_results += f"\nOverall success rate: {((self.total_evaluations - self.failed_evaluations) / self.total_evaluations) * 100:.1f}%"

                print(f"\nBO: {iteration_results}")

                # Show best solutions (only successful ones)
                if len(self.pareto_front) > 0:
                    print(f"\nBO: Current Pareto Front:")
                    for i, (params, objectives) in enumerate(zip(self.pareto_front, self.pareto_objectives)):
                        result_line = f"  Solution {i + 1}: Mech={objectives[0]:.6f}, Vol={objectives[1]:.6f}"
                        print(result_line)
                        iteration_results += f"\n{result_line}"

                        # Show calculated LZ0 and lK value
                        if self.fixed_params:
                            lF_val = params[3]  # lF is at index 3
                            LKG_val = params[2]  # LKG is at index 2
                            LZ0_calc = self.fixed_params.get('LZ', 21.358) - lF_val
                            lK_val = (
                                self.fixed_params.get('standard_LK', 70.0)
                                + (self.fixed_params.get('standard_LKG', 51.62) - LKG_val)
                            )
                            calc_info = f"    LZ0={LZ0_calc:.6f}, lK={lK_val:.6f}"
                            print(calc_info)
                            iteration_results += f"\n{calc_info}"

                        # Verify constraints
                        if not self.validate_individual(np.array(params)):
                            constraint_warning = f"    ⚠️ Solution {i + 1} violates constraints!"
                            print(constraint_warning)
                            iteration_results += f"\n{constraint_warning}"
                else:
                    no_solutions_msg = "No successful solutions found yet - all evaluations failed"
                    print(f"BO: {no_solutions_msg}")
                    iteration_results += f"\n{no_solutions_msg}"

                if self.progress_callback:
                    self.progress_callback.update_results(iteration_results)
                    self.progress_callback.update_progress(current_evaluation, total_evaluations, iteration)
            else:
                no_pareto_msg = f"BO Iteration {iteration} completed - No Pareto front (all evaluations failed)"
                print(f"BO: {no_pareto_msg}")
                if self.progress_callback:
                    self.progress_callback.update_results(no_pareto_msg)

        # Final results and diagnostics
        print(f"\n{'='*80}")
        print(f"BAYESIAN OPTIMIZATION COMPLETED")
        print(f"{'='*80}")

        total_successful = self.total_evaluations - self.failed_evaluations
        overall_success_rate = (total_successful / self.total_evaluations) * 100 if self.total_evaluations > 0 else 0

        final_summary = f"Bayesian Optimization completed:"
        final_summary += f"\nTotal evaluations: {self.total_evaluations}"
        final_summary += f"\nSuccessful evaluations: {total_successful}"
        final_summary += f"\nFailed evaluations: {self.failed_evaluations}"
        final_summary += f"\nOverall success rate: {overall_success_rate:.1f}%"
        final_summary += f"\nFinal Pareto front size: {len(self.pareto_front)}"
        final_summary += f"\nMode: BATCH PARALLEL"

        print(final_summary)

        if self.progress_callback:
            self.progress_callback.update_results(final_summary)

        # Provide diagnostic information if no successful evaluations
        if total_successful == 0:
            diagnostic_msg = "\n❌ OPTIMIZATION FAILED - NO SUCCESSFUL EVALUATIONS"
            diagnostic_msg += "\n\nDiagnostic Information:"
            diagnostic_msg += f"\n- Total attempts: {self.total_evaluations}"
            diagnostic_msg += f"\n- All evaluations returned penalty values (1e6)"
            diagnostic_msg += f"\n- Base folder: {self.optimizer.base_folder}"
            diagnostic_msg += f"\n- Optimization folder: {self.optimizer.algorithm_folder}"
            diagnostic_msg += "\n\nPossible causes:"
            diagnostic_msg += "\n1. fsti_gap.exe not executing properly"
            diagnostic_msg += "\n2. Input files missing or incorrectly formatted"
            diagnostic_msg += "\n3. Parameter values causing simulation instability"
            diagnostic_msg += "\n4. Path or permission issues"
            diagnostic_msg += "\n5. Constraint violations in input generation"
            diagnostic_msg += "\n\nRecommendations:"
            diagnostic_msg += "\n1. Test fsti_gap.exe manually with known good parameters"
            diagnostic_msg += "\n2. Check input file templates"
            diagnostic_msg += "\n3. Verify parameter bounds are reasonable"
            diagnostic_msg += "\n4. Check file permissions and paths"
            diagnostic_msg += "\n5. Review constraint settings"

            print(diagnostic_msg)
            if self.progress_callback:
                self.progress_callback.update_results(diagnostic_msg)

        if self.pareto_front:
            best_individuals = []
            best_objectives = []

            final_validation = "\nFinal constraint validation:"
            for i, (params, objectives) in enumerate(zip(self.pareto_front, self.pareto_objectives)):
                is_valid = self.validate_individual(np.array(params))
                validation_result = f"  Solution {i + 1}: {'✓ Valid' if is_valid else '✗ Invalid'}"
                print(validation_result)
                final_validation += f"\n{validation_result}"

                best_individuals.append(params)
                best_objectives.append(objectives)

            if self.progress_callback:
                self.progress_callback.update_results(final_validation)

            print(f"✓ Returning {len(best_individuals)} successful solutions")
            return best_individuals, np.array(best_objectives)
        else:
            print("✗ No successful solutions to return")
            return [], np.array([])