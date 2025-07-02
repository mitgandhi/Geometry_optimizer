import os
import shutil
import subprocess
import time
import pandas as pd
from pathlib import Path
import platform
import psutil
from cpu_affinity import (
    set_multicore_affinity_windows,
    set_multicore_affinity_psutil,
    monitor_process_affinity
)


class PistonOptimizer:
    """Piston optimization class for running simulations without timeout limits"""

    def __init__(self, base_folder, progress_callback=None):
        self.base_folder = Path(base_folder)
        self.exe_name = "fsti_gap.exe"
        self.nsga3_folder = self.base_folder / "nsga3"  # Legacy compatibility
        self.algorithm_folder = None  # Will be set in setup_optimization_folder
        self.progress_callback = progress_callback

    def setup_optimization_folder(self, algorithm_type="Simple"):
        """Setup the main optimization folder with algorithm-specific naming"""
        try:
            # Create algorithm-specific folder name
            if algorithm_type.lower() == "bayesian":
                self.algorithm_folder = self.base_folder / "bayesian_optimization"
            elif algorithm_type.lower() == "advanced":
                self.algorithm_folder = self.base_folder / "advanced_nsga3"
            else:  # Simple
                self.algorithm_folder = self.base_folder / "simple_nsga3"

            if self.algorithm_folder.exists():
                print(f"Removing existing optimization folder: {self.algorithm_folder}")
                # On Windows, sometimes files are locked, so try multiple times
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        shutil.rmtree(self.algorithm_folder)
                        break
                    except (PermissionError, OSError) as e:
                        if attempt == max_attempts - 1:
                            # Last attempt failed, try alternative approach
                            print(f"Warning: Could not remove folder completely: {e}")
                            # Create a timestamped folder instead
                            import datetime
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            if algorithm_type.lower() == "bayesian":
                                self.algorithm_folder = self.base_folder / f"bayesian_optimization_{timestamp}"
                            elif algorithm_type.lower() == "advanced":
                                self.algorithm_folder = self.base_folder / f"advanced_nsga3_{timestamp}"
                            else:
                                self.algorithm_folder = self.base_folder / f"simple_nsga3_{timestamp}"
                            print(f"Using alternative folder: {self.algorithm_folder}")
                            break
                        else:
                            print(f"Attempt {attempt + 1} failed, retrying in 1 second...")
                            time.sleep(1)

            # Create the folder
            self.algorithm_folder.mkdir(parents=True, exist_ok=True)
            print(f"✓ Optimization folder created: {self.algorithm_folder}")

            # For legacy compatibility, also set nsga3_folder
            self.nsga3_folder = self.algorithm_folder

        except Exception as e:
            print(f"Error setting up optimization folder: {e}")
            # Try with timestamp as fallback
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if algorithm_type.lower() == "bayesian":
                self.algorithm_folder = self.base_folder / f"bayesian_optimization_{timestamp}"
            elif algorithm_type.lower() == "advanced":
                self.algorithm_folder = self.base_folder / f"advanced_nsga3_{timestamp}"
            else:
                self.algorithm_folder = self.base_folder / f"simple_nsga3_{timestamp}"
            try:
                self.algorithm_folder.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created fallback optimization folder: {self.algorithm_folder}")
                self.nsga3_folder = self.algorithm_folder
            except Exception as e2:
                raise RuntimeError(f"Could not create optimization folder: {e2}")

    def create_simulation_folder(self, param_name, generation_or_iteration=None, algorithm_type="Simple"):
        """Create simulation folder with algorithm-specific structure"""

        # Create generation/iteration subfolder
        if generation_or_iteration is not None:
            if algorithm_type.lower() == "bayesian":
                if generation_or_iteration == 0:
                    subfolder_name = "Initial_Sampling"
                else:
                    subfolder_name = f"Iteration_I{generation_or_iteration}"
            else:  # NSGA-III variants
                subfolder_name = f"Generation_G{generation_or_iteration}"

            generation_folder = self.algorithm_folder / subfolder_name
            generation_folder.mkdir(parents=True, exist_ok=True)
            sim_folder = generation_folder / param_name
        else:
            sim_folder = self.algorithm_folder / param_name

        # Ensure the parent folder exists
        if not self.algorithm_folder.exists():
            self.setup_optimization_folder(algorithm_type)

        # Create simulation folder
        try:
            sim_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create simulation folder {sim_folder}: {e}")
            # Try with a shortened name if path is too long
            short_name = param_name[:50] if len(param_name) > 50 else param_name
            if generation_or_iteration is not None:
                if algorithm_type.lower() == "bayesian":
                    if generation_or_iteration == 0:
                        subfolder_name = "Initial_Sampling"
                    else:
                        subfolder_name = f"Iteration_I{generation_or_iteration}"
                else:
                    subfolder_name = f"Generation_G{generation_or_iteration}"
                generation_folder = self.algorithm_folder / subfolder_name
                sim_folder = generation_folder / short_name
            else:
                sim_folder = self.algorithm_folder / short_name
            sim_folder.mkdir(parents=True, exist_ok=True)

        # Copy input files
        input_src = self.base_folder / 'input'
        if input_src.exists():
            try:
                shutil.copytree(input_src, sim_folder / 'input', dirs_exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not copy input files to {sim_folder}: {e}")
                # Try copying files individually
                input_dest = sim_folder / 'input'
                input_dest.mkdir(exist_ok=True)
                for item in input_src.iterdir():
                    if item.is_file():
                        try:
                            shutil.copy2(item, input_dest)
                        except Exception as e2:
                            print(f"Warning: Could not copy {item.name}: {e2}")

        # Copy exe and dlls
        for item in self.base_folder.iterdir():
            if item.is_file() and item.suffix in ['.exe', '.dll']:
                try:
                    shutil.copy2(item, sim_folder)
                except Exception as e:
                    print(f"Warning: Could not copy {item.name}: {e}")

        return sim_folder

    def modify_input_files(self, sim_folder, dK, dZ, LKG, lF, zeta, fixed_params=None):
        """
        Modify input files with new parameter values including LZ0 and LKG calculations

        Args:
            sim_folder: Simulation folder path
            dK, dZ, LKG, lF, zeta: Optimization parameters
            fixed_params: Dictionary containing fixed parameters:
                - LZ: Fixed value for LZ0 calculation
                - standard_LKG: Reference LKG value
                - standard_LK: Reference LK value
        """
        geo_path = sim_folder / 'input' / 'geometry.txt'
        opt_path = sim_folder / 'input' / 'options_piston.txt'

        # Set default fixed parameters if not provided
        if fixed_params is None:
            fixed_params = {
                'LZ': 21.358,  # Default value from geometry.txt
                'standard_LKG': 51.62,  # Default reference value
                'standard_LK': 70.0,  # Default reference LK
                'standard_LSK': 0.0,  # Default CG distance
                'lsk_slope': 0.0      # Default slope for CG shift
            }

        # Calculate LZ0 = LZ - lF
        LZ0_calculated = fixed_params.get('LZ', 21.358) - lF

        # Derive lK from provided LKG value
        lK_calculated = (
            fixed_params.get('standard_LK', 70.0)
            - (fixed_params.get('standard_LKG', 51.62) - LKG)
        )

        # Calculate lSK (center of gravity distance)
        lSK_calculated = (
            fixed_params.get('standard_LSK', 0.0)
            - fixed_params.get('lsk_slope', 0.0)
            * (fixed_params.get('standard_LK', 70.0) - lK_calculated)
        )

        print(f"Calculations for {sim_folder.name}:")
        print(f"  LZ0 = {fixed_params.get('LZ', 21.358)} - {lF} = {LZ0_calculated:.6f}")
        print(
            f"  lK = {fixed_params.get('standard_LK', 70.0)} - ({fixed_params.get('standard_LKG', 51.62)} - {LKG}) = {lK_calculated:.6f}")
        print(
            f"  lSK = {fixed_params.get('standard_LSK', 0.0)} - {fixed_params.get('lsk_slope', 0.0)} * ({fixed_params.get('standard_LK', 70.0)} - {lK_calculated:.6f}) = {lSK_calculated:.6f}")

        # Update geometry parameters
        try:
            # Original parameters
            self._replace_in_file(geo_path, 'dK', f"{dK:.6f}")
            self._replace_in_file(geo_path, 'dZ', f"{dZ:.6f}")
            self._replace_in_file(geo_path, 'lK', f"{lK_calculated:.6f}")
            self._replace_in_file(geo_path, 'lF', f"{lF:.6f}")

            # New calculated parameters
            self._replace_in_file(geo_path, 'lZ0', f"{LZ0_calculated:.6f}")
            self._replace_in_file(geo_path, 'lKG', f"{LKG:.6f}")
            self._replace_in_file(geo_path, 'lSK', f"{lSK_calculated:.6f}")

        except Exception as e:
            print(f"Warning: Could not modify geometry file: {e}")

        # Update options parameter
        try:
            self._replace_in_file(opt_path, 'zeta', str(int(zeta)))
        except Exception as e:
            print(f"Warning: Could not modify options file: {e}")

    def _replace_in_file(self, filepath, param, value):
        """Replace parameter value in file with better error handling"""
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            return

        try:
            lines = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = line.strip().split()
                    if tokens and tokens[0] == param:
                        lines.append(f"\t{param}\t{value}\n")
                    else:
                        lines.append(line)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
        except Exception as e:
            print(f"Error modifying file {filepath}: {e}")

    def run_simulation(self, sim_folder, core_id=None):
        """Run simulation without timeout limits - let it run until completion"""
        exe_path = sim_folder / self.exe_name

        if not exe_path.exists():
            raise FileNotFoundError(f"{self.exe_name} not found in {sim_folder}")

        print(f"Running simulation in {sim_folder.name} - NO TIMEOUT (will run until completion)")

        if platform.system() == "Windows":
            # Windows: Set affinity to ALL cores except core 0
            try:
                import win32process
                import win32api
                import win32event
                import win32con

                # Create process in SUSPENDED state
                startupinfo = win32process.STARTUPINFO()
                startupinfo.dwFlags = win32process.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = win32con.SW_HIDE  # Hide window

                creation_flags = (
                        win32process.CREATE_SUSPENDED |
                        win32process.CREATE_NEW_PROCESS_GROUP |
                        win32process.CREATE_NO_WINDOW  # No console window
                )

                process_info = win32process.CreateProcess(
                    None,  # Application name
                    f'"{exe_path}"',  # Command line
                    None,  # Process security attributes
                    None,  # Thread security attributes
                    0,  # Inherit handles
                    creation_flags,  # Creation flags
                    None,  # Environment
                    str(sim_folder),  # Current directory
                    startupinfo  # Startup info
                )

                process_handle = process_info[0]
                thread_handle = process_info[1]
                process_id = process_info[2]

                # Set multi-core affinity (excluding core 0)
                success = set_multicore_affinity_windows(process_handle, process_id)

                if not success:
                    # Clean up and fail
                    win32process.TerminateProcess(process_handle, 1)
                    win32api.CloseHandle(thread_handle)
                    win32api.CloseHandle(process_handle)
                    raise RuntimeError(f"Could not set multi-core affinity for fsti_gap.exe")

                try:
                    # Resume the process
                    print(f"Resuming fsti_gap.exe with multi-core affinity - NO TIMEOUT")
                    win32process.ResumeThread(thread_handle)
                    win32api.CloseHandle(thread_handle)

                    # Monitor without timeout - let it run until completion
                    success = monitor_process_affinity(process_handle, process_id)

                    # Get final exit code
                    exit_code = win32process.GetExitCodeProcess(process_handle)
                    win32api.CloseHandle(process_handle)

                    if exit_code != 0:
                        raise RuntimeError(f"fsti_gap.exe failed with exit code {exit_code}")

                    total_cores = psutil.cpu_count()
                    print(f"✓ fsti_gap.exe completed successfully on cores 1-{total_cores - 1}")

                except Exception as e:
                    print(f"Windows execution error: {e}")
                    raise

            except ImportError:
                print("win32process not available, using psutil method")
                self._run_simulation_psutil_multicore(sim_folder)
            except Exception as e:
                print(f"Windows method failed: {e}, falling back to psutil")
                self._run_simulation_psutil_multicore(sim_folder)
        else:
            # Linux/Mac
            self._run_simulation_psutil_multicore(sim_folder)

    def _run_simulation_psutil_multicore(self, sim_folder):
        """Psutil method using multi-core affinity without timeout"""
        exe_path = sim_folder / self.exe_name

        # Get all cores except core 0
        total_cores = psutil.cpu_count()
        all_cores_except_0 = list(range(1, total_cores))

        print(f"Using psutil method for multi-core affinity: {all_cores_except_0} - NO TIMEOUT")

        try:
            # Start process
            process = subprocess.Popen(
                f'"{exe_path}"',
                cwd=sim_folder,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True
            )

            print(f"Started fsti_gap.exe with PID {process.pid} - will run until completion")

            # Set multi-core affinity (excluding core 0)
            success = set_multicore_affinity_psutil(process.pid)

            if not success:
                process.kill()
                stdout_data, stderr_data = process.communicate()
                process.wait()
                raise RuntimeError(
                    f"Could not set multi-core affinity: {stderr_data}"
                )

            # Monitor process without timeout - wait indefinitely for completion
            print(f"🕐 Waiting for process {process.pid} to complete (no timeout)...")

            # Simple monitoring - check affinity periodically but no timeout
            check_interval = 30  # Check every 30 seconds
            while process.poll() is None:  # While process is running
                try:
                    p = psutil.Process(process.pid)
                    current_affinity = p.cpu_affinity()
                    if 0 in current_affinity:
                        corrected = [c for c in current_affinity if c != 0] or all_cores_except_0
                        p.cpu_affinity(corrected)
                        print(f"🔧 Corrected affinity to exclude core 0")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break  # Process ended
                except Exception as e:
                    print(f"⚠️ Affinity check failed: {e}")

                time.sleep(check_interval)

            # Process finished - read outputs and check exit code
            stdout_data, stderr_data = process.communicate()
            return_code = process.returncode

            if return_code != 0:
                raise RuntimeError(
                    f"fsti_gap.exe failed with code {return_code}: {stderr_data}"
                )

            print(f"✓ fsti_gap.exe completed successfully on cores {all_cores_except_0}")

        except Exception as e:
            try:
                if 'process' in locals():
                    process.kill()
                    stdout_data, stderr_data = process.communicate()
                    process.wait()
            except Exception:
                pass
            raise RuntimeError(f"Error running fsti_gap.exe: {e}")

    def parse_loss(self, sim_folder):
        """Parse mechanical and volumetric losses from simulation output"""
        piston_path = sim_folder / "output" / "piston" / "piston.txt"

        if not piston_path.exists():
            raise FileNotFoundError(f"piston.txt not found at {piston_path}")

        try:
            # Read results
            df = pd.read_csv(piston_path, delimiter='\t')
            df_rev1 = df[df['revolution'] <= 1.0]

            if df_rev1.empty:
                raise ValueError("No data found for revolution <= 1.0")

            # Calculate mean losses
            mech_loss = df_rev1['Mechanical_Power_Loss'].mean()
            vol_loss = df_rev1['Volumetric_Power_Loss'].mean()

            if pd.isna(mech_loss) or pd.isna(vol_loss):
                raise ValueError("NaN values in loss calculations")

            return abs(mech_loss), abs(vol_loss)

        except Exception as e:
            print(f"Error parsing results from {piston_path}: {e}")
            raise

    def evaluate_individual(self, individual_id, dK, dZ, LKG, lF, zeta, generation=1, core_id=None,
                            algorithm_type="Simple", fixed_params=None):
        """Evaluate a single individual without timeout limits"""
        param_name = f"ind{individual_id}_dK{dK:.3f}_dZ{dZ:.3f}_LKG{LKG:.1f}_lF{lF:.1f}_zeta{int(zeta)}"

        # Report current evaluation
        if self.progress_callback:
            self.progress_callback.update_current_evaluation(individual_id, [dK, dZ, LKG, lF, zeta])

        try:
            sim_folder = self.create_simulation_folder(param_name, generation, algorithm_type)

            # Pass fixed_params to modify_input_files for LZ0 and lK calculations
            self.modify_input_files(sim_folder, dK, dZ, LKG, lF, int(zeta), fixed_params)

            # Run simulation without timeout - will run until completion
            self.run_simulation(sim_folder)

            mech_loss, vol_loss = self.parse_loss(sim_folder)

            print(f"✓ Individual {individual_id}: Mechanical={mech_loss:.3f}, Volumetric={vol_loss:.3f}")

            # Report result
            if self.progress_callback:
                self.progress_callback.update_results(
                    f"Individual {individual_id}: Mech={mech_loss:.3f}, Vol={vol_loss:.3f}")

            return mech_loss, vol_loss

        except Exception as e:
            print(f"✗ Individual {individual_id} failed: {e}")

            # Report failure
            if self.progress_callback:
                self.progress_callback.update_results(f"Individual {individual_id}: FAILED - {str(e)}")

            return 1e6, 1e6  # Penalty values

    def cleanup_simulation_folder(self, param_name):
        """Clean up simulation folder to save disk space"""
        sim_folder = self.algorithm_folder / param_name
        if sim_folder.exists():
            # Keep only essential result files
            essential_files = ['piston.txt']

            try:
                output_folder = sim_folder / "output"
                if output_folder.exists():
                    for item in output_folder.rglob("*"):
                        if item.is_file() and item.name not in essential_files:
                            try:
                                item.unlink()
                            except Exception as e:
                                print(f"Warning: Could not delete {item}: {e}")

                # Remove input folder after simulation
                input_folder = sim_folder / "input"
                if input_folder.exists():
                    try:
                        shutil.rmtree(input_folder)
                    except Exception as e:
                        print(f"Warning: Could not remove input folder {input_folder}: {e}")

                print(f"Cleaned up {param_name}")

            except Exception as e:
                print(f"Warning: Could not clean up {param_name}: {e}")

    def save_results(self, best_individuals, best_objectives, output_file=None, algorithm_type="Simple"):
        """Save optimization results to file"""
        if output_file is None:
            output_file = self.algorithm_folder / "optimization_results.txt"

        param_names = ['dK', 'dZ', 'LKG', 'lF', 'zeta']

        try:
            with open(output_file, 'w') as f:
                f.write(f"{algorithm_type} Piston Optimization Results\n")
                f.write("=" * 50 + "\n\n")

                if best_individuals:
                    f.write(f"Found {len(best_individuals)} Pareto optimal solutions:\n\n")

                    for i, (individual, objectives) in enumerate(zip(best_individuals, best_objectives)):
                        f.write(f"Solution {i + 1}:\n")
                        f.write(f"  Mechanical Loss: {objectives[0]:.6f}\n")
                        f.write(f"  Volumetric Loss: {objectives[1]:.6f}\n")
                        f.write("  Parameters:\n")

                        for j, param in enumerate(param_names):
                            if param == 'zeta':
                                f.write(f"    {param}: {int(individual[j])}\n")
                            else:
                                f.write(f"    {param}: {individual[j]:.6f}\n")
                        f.write("\n")

                else:
                    f.write("No valid solutions found!\n")

            print(f"Results saved to {output_file}")

        except Exception as e:
            print(f"Error saving results: {e}")

    def generate_summary_report(self, best_individuals, best_objectives, config):
        """Generate a comprehensive summary report"""
        algorithm_type = config.get('algorithm_type', 'Simple')
        report_file = self.algorithm_folder / f"{algorithm_type.lower()}_optimization_summary.txt"

        try:
            with open(report_file, 'w') as f:
                f.write(f"{algorithm_type} Piston Optimization Summary Report\n")
                f.write("=" * 60 + "\n\n")

                # Configuration summary
                f.write("OPTIMIZATION CONFIGURATION:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Algorithm: {algorithm_type}\n")

                if algorithm_type.lower() == "bayesian":
                    f.write(f"Initial Samples: {config.get('initial_samples', 'Unknown')}\n")
                    f.write(f"BO Iterations: {config.get('bo_iterations', 'Unknown')}\n")
                    f.write(f"Acquisition Function: {config.get('acquisition_function', 'Unknown')}\n")
                    f.write(
                        f"Total Evaluations: ~{config.get('initial_samples', 0) + config.get('bo_iterations', 0)}\n")
                else:
                    f.write(f"Population Size: {config.get('population_size', 'Unknown')}\n")
                    f.write(f"Generations: {config.get('generations', 'Unknown')}\n")
                    f.write(f"Total Evaluations: {config.get('population_size', 0) * config.get('generations', 0)}\n")

                    if algorithm_type.lower() == "advanced":
                        f.write(f"Reference Partitions: {config.get('reference_partitions', 'Unknown')}\n")

                f.write(f"Base Folder: {config.get('base_folder', 'Unknown')}\n")
                f.write(f"Optimization Folder: {self.algorithm_folder}\n")
                f.write(f"Timeout Policy: NO TIMEOUT (run until completion)\n\n")

                # Fixed parameters for calculations
                fixed_params = config.get('fixed_params', {})
                if fixed_params:
                    f.write("FIXED PARAMETERS FOR CALCULATIONS:\n")
                    f.write("-" * 35 + "\n")
                    f.write(f"LZ (for LZ0 calculation): {fixed_params.get('LZ', 'Not set')}\n")
                    f.write(f"Standard LKG: {fixed_params.get('standard_LKG', 'Not set')}\n")
                    f.write(f"Standard LK: {fixed_params.get('standard_LK', 'Not set')}\n")
                    f.write("Calculations:\n")
                    f.write("  LZ0 = LZ - lF\n")
                    f.write("  lK = standard_LK - (standard_LKG - LKG)\n\n")

                # Constraint information
                constraints_config = config.get('constraints', {})
                if constraints_config:
                    f.write("CONSTRAINTS:\n")
                    f.write("-" * 15 + "\n")
                    for name, constraint_config in constraints_config.items():
                        status = "Enabled" if constraint_config.get('active', False) else "Disabled"
                        f.write(f"{name}: {status}\n")
                        if 'min_difference' in constraint_config and constraint_config.get('active', False):
                            f.write(
                                f"  Range: [{constraint_config['min_difference']}, {constraint_config['max_difference']}]\n")
                    f.write("\n")

                # Parameter bounds
                f.write("PARAMETER BOUNDS:\n")
                f.write("-" * 20 + "\n")
                param_bounds = config.get('param_bounds', {})
                for param, bounds in param_bounds.items():
                    f.write(f"{param}: [{bounds['min']:.3f}, {bounds['max']:.3f}]\n")
                f.write("\n")

                # System information
                f.write("SYSTEM INFORMATION:\n")
                f.write("-" * 20 + "\n")
                total_cores = psutil.cpu_count()
                f.write(f"Total CPU Cores: {total_cores}\n")
                f.write(f"Cores Used: 1-{total_cores - 1} (Core 0 reserved)\n")
                f.write(f"Available Memory: {psutil.virtual_memory().total / (1024 ** 3):.1f} GB\n")
                f.write(f"Timeout Policy: NO TIMEOUT - simulations run until completion\n\n")

                # Results summary
                f.write("OPTIMIZATION RESULTS:\n")
                f.write("-" * 25 + "\n")

                if best_individuals:
                    f.write(f"Number of Pareto Optimal Solutions: {len(best_individuals)}\n\n")

                    # Best solution for each objective
                    mech_losses = [obj[0] for obj in best_objectives]
                    vol_losses = [obj[1] for obj in best_objectives]

                    best_mech_idx = mech_losses.index(min(mech_losses))
                    best_vol_idx = vol_losses.index(min(vol_losses))

                    f.write(f"Best Mechanical Loss: {min(mech_losses):.6f}\n")
                    f.write(f"Best Volumetric Loss: {min(vol_losses):.6f}\n\n")

                    # Detailed results
                    f.write("DETAILED PARETO FRONT:\n")
                    f.write("-" * 25 + "\n")

                    param_names = ['dK', 'dZ', 'LKG', 'lF', 'zeta']

                    for i, (individual, objectives) in enumerate(zip(best_individuals, best_objectives)):
                        f.write(f"\nSolution {i + 1}:\n")
                        f.write(f"  Mechanical Loss: {objectives[0]:.6f}\n")
                        f.write(f"  Volumetric Loss: {objectives[1]:.6f}\n")
                        f.write("  Parameters: ")

                        param_str = []
                        for j, param in enumerate(param_names):
                            if param == 'zeta':
                                param_str.append(f"{param}={int(individual[j])}")
                            else:
                                param_str.append(f"{param}={individual[j]:.3f}")

                        f.write(", ".join(param_str) + "\n")

                        # Show calculated values
                        if fixed_params:
                            lF_val = individual[3]  # lF is at index 3
                            LKG_val = individual[2]  # LKG is at index 2
                            LZ0_calc = fixed_params.get('LZ', 21.358) - lF_val
                            lK_val = (
                                fixed_params.get('standard_LK', 70.0)
                                - (fixed_params.get('standard_LKG', 51.62) - LKG_val)
                            )
                            f.write(f"  Calculated LZ0: {LZ0_calc:.6f}\n")
                            f.write(f"  Calculated lK: {lK_val:.6f}\n")

                else:
                    f.write("No valid solutions found!\n")
                    f.write("Please check:\n")
                    f.write("- Simulation setup and files\n")
                    f.write("- Parameter bounds and constraints\n")
                    f.write("- System resources\n")
                    f.write("- fsti_gap.exe execution permissions\n")

            print(f"Summary report saved to {report_file}")
            return report_file

        except Exception as e:
            print(f"Error generating summary report: {e}")
            return None