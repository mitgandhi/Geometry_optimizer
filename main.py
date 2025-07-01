#!/usr/bin/env python3
"""
NSGA-III Piston Optimization - Main Runner - No Timeout Version
===============================================================

This is the main entry point for the NSGA-III piston optimization system.
It integrates the GUI, CPU affinity management, NSGA-III algorithm, constraint
handling, and piston simulation optimizer.

NO TIMEOUT LIMITS - simulations will run until natural completion.

Usage:
    python main.py                    # Run with GUI
    python main.py --config file.json # Run with config file
    python main.py --cli              # Run with command line interface
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
import psutil

# Import our modules
from gui_interface import run_gui
from cpu_affinity import get_available_cores, force_exclude_core0
from nsga3_algorithm import SimpleNSGA3, AdvancedNSGA3
from bayesian_optimization import BayesianOptimization
from piston_optimizer import PistonOptimizer
from constraints import ConstraintManager


def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    NSGA-III Piston Optimizer                ║
║                       NO TIMEOUT VERSION                    ║
║                                                              ║
║  Multi-Objective Optimization for Piston Design Parameters  ║
║                       with Constraints                      ║
║                                                              ║
║  Features:                                                   ║
║  • Simple & Advanced NSGA-III algorithms                    ║
║  • Bayesian Optimization for expensive simulations         ║
║  • Customizable parameter constraints                       ║
║  • Multi-core CPU affinity management                       ║
║  • Interactive GUI configuration                            ║
║  • Automatic result analysis                                ║
║  • NO TIMEOUT - simulations run until completion           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def validate_config(config):
    """Validate configuration parameters"""
    errors = []

    # Required fields
    required_fields = ['base_folder', 'population_size', 'generations', 'param_bounds']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Check base folder
    if 'base_folder' in config:
        base_path = Path(config['base_folder'])
        if not base_path.exists():
            errors.append(f"Base folder does not exist: {config['base_folder']}")
        elif not (base_path / 'fsti_gap.exe').exists():
            errors.append(f"fsti_gap.exe not found in base folder: {config['base_folder']}")
        elif not (base_path / 'input').exists():
            errors.append(f"Input folder not found in base folder: {config['base_folder']}")

    # Check numeric parameters
    if 'population_size' in config:
        if not isinstance(config['population_size'], int) or config['population_size'] < 4:
            errors.append("Population size must be an integer >= 4")

    if 'generations' in config:
        if not isinstance(config['generations'], int) or config['generations'] < 1:
            errors.append("Generations must be an integer >= 1")

    # Check parameter bounds
    if 'param_bounds' in config:
        required_params = ['dK', 'dZ', 'LKG', 'lF', 'zeta']
        for param in required_params:
            if param not in config['param_bounds']:
                errors.append(f"Missing parameter bounds for: {param}")
            else:
                bounds = config['param_bounds'][param]
                if 'min' not in bounds or 'max' not in bounds:
                    errors.append(f"Invalid bounds format for {param}")
                elif bounds['min'] >= bounds['max']:
                    errors.append(f"Invalid bounds for {param}: min must be < max")

    # Check constraints
    if 'constraints' in config:
        constraints = config['constraints']

        # Check dZ-dK difference constraint
        if 'dZ_dK_difference_range' in constraints:
            dz_dk_config = constraints['dZ_dK_difference_range']
            if dz_dk_config.get('active', False):
                min_diff = dz_dk_config.get('min_difference')
                max_diff = dz_dk_config.get('max_difference')
                if min_diff is not None and max_diff is not None:
                    if min_diff >= max_diff:
                        errors.append("dZ-dK minimum difference must be less than maximum difference")

    return errors


def create_default_config():
    """Create default configuration"""
    return {
        'base_folder': '',
        'population_size': 20,
        'generations': 10,
        'algorithm_type': 'Simple',
        'reference_partitions': 12,
        'eta_c': 20.0,
        'eta_m': 20.0,
        'initial_samples': 10,
        'bo_iterations': 20,
        'acquisition_function': 'ei',
        'param_bounds': {
            'dK': {'min': 19.0, 'max': 20.0},
            'dZ': {'min': 19.2, 'max': 20.0},
            'LKG': {'min': 50.0, 'max': 70.0},
            'lF': {'min': 30.0, 'max': 40.0},
            'zeta': {'min': 3, 'max': 7}
        },
        'constraints': {
            'dK_less_than_dZ': {'active': True},
            'dZ_dK_difference_range': {
                'active': True,
                'min_difference': 0.1,
                'max_difference': 0.8
            }
        }
    }


def load_config_file(filename):
    """Load configuration from JSON file"""
    try:
        with open(filename, 'r') as f:
            config = json.load(f)
        print(f"✓ Configuration loaded from {filename}")
        return config
    except FileNotFoundError:
        print(f"✗ Configuration file not found: {filename}")
        return None
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in configuration file: {e}")
        return None
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return None


def save_config_file(config, filename):
    """Save configuration to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Configuration saved to {filename}")
        return True
    except Exception as e:
        print(f"✗ Error saving configuration: {e}")
        return False


def cli_interface():
    """Command line interface for configuration"""
    print("\n" + "=" * 60)
    print("COMMAND LINE CONFIGURATION")
    print("=" * 60)

    config = create_default_config()

    # Get base folder
    while True:
        base_folder = input(f"\nBase folder [{config['base_folder']}]: ").strip()
        if not base_folder and config['base_folder']:
            break
        elif not base_folder:
            print("Base folder is required!")
            continue

        base_path = Path(base_folder)
        if not base_path.exists():
            print(f"Folder does not exist: {base_folder}")
            continue
        elif not (base_path / 'fsti_gap.exe').exists():
            print(f"fsti_gap.exe not found in: {base_folder}")
            continue
        elif not (base_path / 'input').exists():
            print(f"Input folder not found in: {base_folder}")
            continue

        config['base_folder'] = str(base_path)
        break

    # Get algorithm parameters
    try:
        pop_size = input(f"\nPopulation size [{config['population_size']}]: ").strip()
        if pop_size:
            config['population_size'] = int(pop_size)

        generations = input(f"Generations [{config['generations']}]: ").strip()
        if generations:
            config['generations'] = int(generations)

        algo_type = input(f"Algorithm type (Simple/Advanced/Bayesian) [{config['algorithm_type']}]: ").strip()
        if algo_type and algo_type.lower() in ['simple', 'advanced', 'bayesian']:
            config['algorithm_type'] = algo_type.capitalize()

    except ValueError as e:
        print(f"Invalid input: {e}")
        return None

    # Parameter bounds (simplified)
    print(f"\nUsing default parameter bounds:")
    for param, bounds in config['param_bounds'].items():
        print(f"  {param}: [{bounds['min']}, {bounds['max']}]")

    modify_bounds = input("\nModify parameter bounds? (y/n) [n]: ").strip().lower()
    if modify_bounds == 'y':
        for param in config['param_bounds'].keys():
            try:
                bounds_str = input(f"{param} bounds [min,max]: ").strip()
                if bounds_str:
                    min_val, max_val = map(float, bounds_str.split(','))
                    config['param_bounds'][param]['min'] = min_val
                    config['param_bounds'][param]['max'] = max_val
            except ValueError:
                print(f"Invalid bounds for {param}, keeping default")

    # Constraints configuration
    print(f"\nConstraint Configuration:")
    print(f"Current constraints:")
    for name, constraint_config in config['constraints'].items():
        status = "Enabled" if constraint_config.get('active', False) else "Disabled"
        print(f"  {name}: {status}")
        if 'min_difference' in constraint_config:
            print(f"    Range: [{constraint_config['min_difference']}, {constraint_config['max_difference']}]")

    modify_constraints = input("\nModify constraints? (y/n) [n]: ").strip().lower()
    if modify_constraints == 'y':
        # dK < dZ constraint
        dk_dz_active = input("Enable dK < dZ constraint? (y/n) [y]: ").strip().lower()
        config['constraints']['dK_less_than_dZ']['active'] = dk_dz_active != 'n'

        # dZ - dK difference constraint
        dz_dk_active = input("Enable dZ - dK difference constraint? (y/n) [y]: ").strip().lower()
        config['constraints']['dZ_dK_difference_range']['active'] = dz_dk_active != 'n'

        if config['constraints']['dZ_dK_difference_range']['active']:
            try:
                range_str = input("dZ - dK difference range [min,max] [0.1,0.8]: ").strip()
                if range_str:
                    min_diff, max_diff = map(float, range_str.split(','))
                    config['constraints']['dZ_dK_difference_range']['min_difference'] = min_diff
                    config['constraints']['dZ_dK_difference_range']['max_difference'] = max_diff
            except ValueError:
                print("Invalid range, keeping default")

    return config


def test_constraints_with_bounds(config):
    """Test constraints with current parameter bounds"""
    print("\n" + "=" * 60)
    print("CONSTRAINT VALIDATION")
    print("=" * 60)

    # Create constraint manager and configure it
    cm = ConstraintManager()

    constraints_config = config.get('constraints', {})

    # Configure dK < dZ constraint
    dk_dz_config = constraints_config.get('dK_less_than_dZ', {})
    if not dk_dz_config.get('active', True):
        cm.deactivate_constraint('dK_less_than_dZ')

    # Configure dZ - dK difference constraint
    dz_dk_config = constraints_config.get('dZ_dK_difference_range', {})
    if not dz_dk_config.get('active', True):
        cm.deactivate_constraint('dZ_dK_difference_range')
    else:
        constraint_config = {
            'min_difference': dz_dk_config.get('min_difference', 0.1),
            'max_difference': dz_dk_config.get('max_difference', 0.8)
        }
        cm.set_constraint_config('dZ_dK_difference_range', constraint_config)

    print("Active constraints:")
    cm.list_constraints()

    # Test parameter generation
    param_bounds = config['param_bounds']
    print("Testing parameter generation...")

    valid_params = cm.generate_valid_parameters(param_bounds)
    if valid_params:
        param_names = ['dK', 'dZ', 'LKG', 'lF', 'zeta']
        print("✓ Generated valid parameters:")
        for i, param in enumerate(param_names):
            if param == 'zeta':
                print(f"  {param}: {int(valid_params[i])}")
            else:
                print(f"  {param}: {valid_params[i]:.3f}")
    else:
        print("✗ Could not generate valid parameters!")
        return False

    return True


def run_optimization(config):
    """Run the optimization with given configuration - NO TIMEOUT"""
    print("\n" + "=" * 60)
    print("STARTING OPTIMIZATION - NO TIMEOUT")
    print("=" * 60)

    # Force exclude core 0 from main process
    force_exclude_core0()

    # Get available cores
    available_cores = get_available_cores()
    total_cores = psutil.cpu_count()

    print(f"\nSYSTEM CONFIGURATION:")
    print(f"  Total cores: {total_cores}")
    print(f"  Core 0: RESERVED for system")
    print(f"  Available cores: {available_cores}")
    print(f"  Parallel workers: {len(available_cores)}")
    print(f"  TIMEOUT POLICY: NO TIMEOUT - simulations run until completion")

    print(f"\nOPTIMIZATION PARAMETERS:")
    print(f"  Algorithm: {config['algorithm_type']}")

    if config['algorithm_type'] == 'Bayesian':
        print(f"  Initial Samples: {config.get('initial_samples', 10)}")
        print(f"  BO Iterations: {config.get('bo_iterations', 20)}")
        print(f"  Acquisition Function: {config.get('acquisition_function', 'ei').upper()}")
        print(f"  Total Evaluations: ~{config.get('initial_samples', 10) + config.get('bo_iterations', 20)}")
    else:
        print(f"  Population size: {config['population_size']}")
        print(f"  Generations: {config['generations']}")

    print(f"  Base folder: {config['base_folder']}")

    # Display constraints
    constraints_config = config.get('constraints', {})
    print(f"\nCONSTRAINTS:")
    for name, constraint_config in constraints_config.items():
        status = "✓ Enabled" if constraint_config.get('active', False) else "✗ Disabled"
        print(f"  {status}: {name}")
        if 'min_difference' in constraint_config and constraint_config.get('active', False):
            print(f"    Range: [{constraint_config['min_difference']}, {constraint_config['max_difference']}]")

    # Convert parameter bounds to expected format
    param_bounds = {}
    for param, bounds in config['param_bounds'].items():
        param_bounds[param] = (bounds['min'], bounds['max'])

    # Create optimizer
    optimizer = PistonOptimizer(config['base_folder'])

    # Choose algorithm
    if config.get('algorithm_type', 'Simple').lower() == 'bayesian':
        print(f"  Acquisition function: {config.get('acquisition_function', 'ei')}")
        print(f"  Batch size: {config.get('batch_size', 5)} individuals per iteration")

        optimizer_instance = BayesianOptimization(
            optimizer=optimizer,
            param_bounds=param_bounds,
            available_cores=available_cores,
            n_initial=config.get('initial_samples', 10),
            n_iterations=config.get('bo_iterations', 20),
            acquisition=config.get('acquisition_function', 'ei'),
            constraints_config=constraints_config,
            fixed_params=config.get('fixed_params', {}),
            batch_size=config.get('batch_size', 5)  # NEW: Pass batch size
        )
    elif config.get('algorithm_type', 'Simple').lower() == 'advanced':
        print(f"  Reference partitions: {config.get('reference_partitions', 12)}")
        optimizer_instance = AdvancedNSGA3(
            optimizer=optimizer,
            param_bounds=param_bounds,
            available_cores=available_cores,
            pop_size=config['population_size'],
            generations=config['generations'],
            n_partitions=config.get('reference_partitions', 12),
            constraints_config=constraints_config,
            fixed_params=config.get('fixed_params', {})
        )
    else:
        optimizer_instance = SimpleNSGA3(
            optimizer=optimizer,
            param_bounds=param_bounds,
            available_cores=available_cores,
            pop_size=config['population_size'],
            generations=config['generations'],
            constraints_config=constraints_config,
            fixed_params=config.get('fixed_params', {})
        )

    # Start timing
    start_time = time.time()

    try:
        print(f"\n🚀 Starting optimization with NO TIMEOUT LIMITS")
        print(f"Each simulation will run until natural completion")

        # Run optimization
        best_individuals, best_objectives = optimizer_instance.run_optimization()

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60

        print(f"\n" + "=" * 60)
        print("OPTIMIZATION COMPLETED")
        print(f"=" * 60)
        print(f"Total time: {hours:02d}:{minutes:02d}:{seconds:05.2f}")

        if best_individuals:
            print(f"Found {len(best_individuals)} Pareto optimal solutions:")
            param_names = ['dK', 'dZ', 'LKG', 'lF', 'zeta']

            # Final constraint validation
            cm = ConstraintManager()
            # Configure constraints
            dk_dz_config = constraints_config.get('dK_less_than_dZ', {})
            if not dk_dz_config.get('active', True):
                cm.deactivate_constraint('dK_less_than_dZ')

            dz_dk_config = constraints_config.get('dZ_dK_difference_range', {})
            if not dz_dk_config.get('active', True):
                cm.deactivate_constraint('dZ_dK_difference_range')
            else:
                constraint_config = {
                    'min_difference': dz_dk_config.get('min_difference', 0.1),
                    'max_difference': dz_dk_config.get('max_difference', 0.8)
                }
                cm.set_constraint_config('dZ_dK_difference_range', constraint_config)

            for i, (individual, objectives) in enumerate(zip(best_individuals, best_objectives)):
                print(f"\nSolution {i + 1}: Mechanical={objectives[0]:.6f}, Volumetric={objectives[1]:.6f}")
                param_str = []
                param_dict = {}
                for j, param in enumerate(param_names):
                    if param == 'zeta':
                        param_str.append(f"{param}={int(individual[j])}")
                        param_dict[param] = int(individual[j])
                    else:
                        param_str.append(f"{param}={individual[j]:.3f}")
                        param_dict[param] = individual[j]
                print(f"  Parameters: {', '.join(param_str)}")

                # Validate constraints
                is_valid = cm.validate_parameters(**param_dict)
                if is_valid:
                    print(f"  Constraints: ✓ Valid")
                else:
                    violated = cm.get_violated_constraints(**param_dict)
                    print(f"  Constraints: ✗ Violated - {', '.join([v['name'] for v in violated])}")

            # Save results
            optimizer.save_results(best_individuals, best_objectives)

            # Update config with constraint information for the report
            enhanced_config = config.copy()
            enhanced_config['constraint_validation'] = 'Enabled'
            enhanced_config['total_constraints'] = len(
                [c for c in constraints_config.values() if c.get('active', False)])
            enhanced_config['timeout_policy'] = 'NO TIMEOUT - run until completion'

            report_file = optimizer.generate_summary_report(best_individuals, best_objectives, enhanced_config)

            print(f"\n✓ Results saved to optimization folder")
            print(f"✓ Summary report: {report_file}")
        else:
            print("✗ No valid solutions found!")
            print("Check simulation setup, parameter bounds, constraints, and system resources.")

        return best_individuals, best_objectives

    except KeyboardInterrupt:
        print(f"\n\nOptimization interrupted by user")
        print(f"Elapsed time: {(time.time() - start_time) / 60:.1f} minutes")
        return None, None

    except Exception as e:
        print(f"\n✗ Optimization failed: {e}")
        print(f"Elapsed time: {(time.time() - start_time) / 60:.1f} minutes")
        return None, None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="NSGA-III Piston Optimization System with Constraints - NO TIMEOUT VERSION",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run with GUI
  python main.py --config my.json  # Run with config file
  python main.py --cli             # Run with command line interface
  python main.py --validate conf.json  # Validate configuration file
  python main.py --test-constraints conf.json  # Test constraints

NO TIMEOUT VERSION: All simulations will run until natural completion.
        """
    )

    parser.add_argument('--config', '-c', type=str,
                        help='Load configuration from JSON file')
    parser.add_argument('--cli', action='store_true',
                        help='Use command line interface instead of GUI')
    parser.add_argument('--validate', '-v', type=str,
                        help='Validate configuration file and exit')
    parser.add_argument('--test-constraints', type=str,
                        help='Test constraints with configuration file and exit')
    parser.add_argument('--save-default', type=str,
                        help='Save default configuration to file and exit')
    parser.add_argument('--no-banner', action='store_true',
                        help='Skip banner display')

    args = parser.parse_args()

    # Print banner unless suppressed
    if not args.no_banner:
        print_banner()

    # Handle special commands
    if args.save_default:
        config = create_default_config()
        if save_config_file(config, args.save_default):
            print(f"✓ Default configuration saved to {args.save_default}")
        else:
            print(f"✗ Failed to save default configuration")
        return 0

    if args.validate:
        config = load_config_file(args.validate)
        if config is None:
            return 1

        errors = validate_config(config)
        if errors:
            print(f"✗ Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return 1
        else:
            print(f"✓ Configuration is valid")
            return 0

    if args.test_constraints:
        config = load_config_file(args.test_constraints)
        if config is None:
            return 1

        errors = validate_config(config)
        if errors:
            print(f"✗ Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return 1

        success = test_constraints_with_bounds(config)
        return 0 if success else 1

    # Determine configuration source
    config = None

    if args.config:
        # Load from file
        config = load_config_file(args.config)
        if config is None:
            print("✗ Failed to load configuration file")
            return 1
    elif args.cli:
        # Command line interface
        config = cli_interface()
        if config is None:
            print("✗ Configuration setup cancelled")
            return 1
    else:
        # GUI interface - optimization runs within GUI now
        print("Starting GUI configuration...")
        config = run_gui()  # This will return None unless optimization was started

        if config is None:
            print("GUI closed without starting optimization")
            return 0
        else:
            # If we get here, optimization was already run through the GUI
            print("✅ Optimization completed through GUI interface")
            return 0

    # For CLI and config file modes, continue with validation and optimization
    if config:
        # Validate configuration
        errors = validate_config(config)
        if errors:
            print(f"✗ Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return 1

        # Test constraints if they are enabled
        constraints_config = config.get('constraints', {})
        active_constraints = [name for name, conf in constraints_config.items() if conf.get('active', False)]
        if active_constraints:
            print(f"\nTesting {len(active_constraints)} active constraints...")
            if not test_constraints_with_bounds(config):
                print("✗ Constraint testing failed - check parameter bounds and constraint settings")
                return 1

        # Offer to save configuration for CLI mode
        if args.cli:
            save_config = input("\nSave this configuration? (y/n) [n]: ").strip().lower()
            if save_config == 'y':
                filename = input("Configuration filename [optimization_config.json]: ").strip()
                if not filename:
                    filename = "optimization_config.json"
                save_config_file(config, filename)

        # Run optimization
        try:
            print(f"\n🚀 STARTING OPTIMIZATION WITH NO TIMEOUT LIMITS")
            print(f"Each simulation will run until natural completion")
            print(f"This may take several hours depending on simulation complexity")

            best_individuals, best_objectives = run_optimization(config)

            if best_individuals:
                print(f"\n🎉 Optimization completed successfully!")
                return 0
            else:
                print(f"\n⚠️  Optimization completed without finding valid solutions")
                return 2

        except Exception as e:
            print(f"\n💥 Fatal error during optimization: {e}")
            return 1


def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        'numpy', 'pandas', 'psutil', 'tkinter'
    ]

    # Bayesian Optimization specific dependencies
    bayesian_modules = [
        'scipy', 'sklearn'
    ]

    missing_modules = []
    missing_bayesian = []

    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    for module in bayesian_modules:
        try:
            __import__(module)
        except ImportError:
            missing_bayesian.append(module)

    if missing_modules:
        print(f"✗ Missing required modules: {', '.join(missing_modules)}")
        print("Please install them using: pip install " + ' '.join(missing_modules))
        return False

    if missing_bayesian:
        print(f"⚠️ Missing Bayesian Optimization modules: {', '.join(missing_bayesian)}")
        print("Bayesian Optimization will not be available. Install with: pip install " + ' '.join(missing_bayesian))
        print("You can still use Simple and Advanced NSGA-III algorithms.")
        # Don't return False - allow other algorithms to work

    return True


if __name__ == "__main__":
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check minimum Python version
    if sys.version_info < (3, 7):
        print("✗ Python 3.7 or higher is required")
        sys.exit(1)

    # Check if we have sufficient CPU cores
    if psutil.cpu_count() < 2:
        print("✗ At least 2 CPU cores are required (core 0 is reserved)")
        sys.exit(1)

    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)