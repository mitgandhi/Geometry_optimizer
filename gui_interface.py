import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
import time
from pathlib import Path
import psutil
from constraints import ConstraintManager


class OptimizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NSGA-III Piston Optimization")
        self.root.geometry("950x1100")  # Slightly larger to accommodate new fields

        # Initialize constraint manager
        self.constraint_manager = ConstraintManager()

        # Configuration data with new fixed parameters
        self.config = {
            'base_folder': '',
            'population_size': 20,
            'generations': 10,
            'algorithm_type': 'Simple',
            'param_bounds': {
                'dK': {'min': 19.0, 'max': 20.0},
                'dZ': {'min': 19.2, 'max': 20.0},
                'lK': {'min': 50.0, 'max': 70.0},
                'lF': {'min': 30.0, 'max': 40.0},
                'zeta': {'min': 3, 'max': 7}
            },
            'constraints': {
                'dZ_dK_difference_range': {
                    'min_difference': 0.1,
                    'max_difference': 0.8
                }
            },
            'fixed_params': {
                'LZ': 21.358,
                'longest_gap_length': 51.62,
                'max_lK': 70.0
            }
        }

        # Progress tracking variables
        self.optimization_running = False
        self.start_time = None
        self.optimization_started = False  # Flag to track if optimization was started

        self.create_widgets()
        self.load_default_config()

    def create_widgets(self):
        # Main notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Basic Settings Tab
        basic_frame = ttk.Frame(notebook)
        notebook.add(basic_frame, text="Basic Settings")
        self.create_basic_settings(basic_frame)

        # Parameter Bounds Tab
        bounds_frame = ttk.Frame(notebook)
        notebook.add(bounds_frame, text="Parameter Bounds")
        self.create_parameter_bounds(bounds_frame)

        # Fixed Parameters Tab (NEW)
        fixed_params_frame = ttk.Frame(notebook)
        notebook.add(fixed_params_frame, text="Fixed Parameters")
        self.create_fixed_parameters(fixed_params_frame)

        # Constraints Tab
        constraints_frame = ttk.Frame(notebook)
        notebook.add(constraints_frame, text="Constraints")
        self.create_constraints_tab(constraints_frame)

        # System Info Tab
        system_frame = ttk.Frame(notebook)
        notebook.add(system_frame, text="System Info")
        self.create_system_info(system_frame)

        # Control buttons at bottom
        self.create_control_buttons()

    def create_basic_settings(self, parent):
        # Base folder selection
        folder_frame = ttk.LabelFrame(parent, text="Simulation Files", padding=10)
        folder_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(folder_frame, text="Base Folder:").grid(row=0, column=0, sticky='w', pady=2)
        self.folder_var = tk.StringVar()
        folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var, width=50)
        folder_entry.grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(folder_frame, text="Browse",
                   command=self.browse_folder).grid(row=0, column=2, padx=5, pady=2)

        # Algorithm settings
        algo_frame = ttk.LabelFrame(parent, text="Algorithm Settings", padding=10)
        algo_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(algo_frame, text="Algorithm Type:").grid(row=0, column=0, sticky='w', pady=2)
        self.algo_var = tk.StringVar(value="Simple")
        algo_combo = ttk.Combobox(algo_frame, textvariable=self.algo_var,
                                  values=["Simple", "Advanced", "Bayesian"], state="readonly")
        algo_combo.grid(row=0, column=1, sticky='w', padx=5, pady=2)

        # Bind algorithm change to update GUI
        algo_combo.bind('<<ComboboxSelected>>', self.on_algorithm_change)

        ttk.Label(algo_frame, text="Population Size:").grid(row=1, column=0, sticky='w', pady=2)
        self.pop_var = tk.IntVar(value=20)
        pop_spin = ttk.Spinbox(algo_frame, from_=10, to=100, textvariable=self.pop_var, width=10)
        pop_spin.grid(row=1, column=1, sticky='w', padx=5, pady=2)

        ttk.Label(algo_frame, text="Generations:").grid(row=2, column=0, sticky='w', pady=2)
        self.gen_var = tk.IntVar(value=10)
        gen_spin = ttk.Spinbox(algo_frame, from_=5, to=100, textvariable=self.gen_var, width=10)
        gen_spin.grid(row=2, column=1, sticky='w', padx=5, pady=2)

        # Advanced settings (for Advanced algorithm)
        self.advanced_frame = ttk.LabelFrame(parent, text="Advanced NSGA-III Settings", padding=10)
        self.advanced_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(self.advanced_frame, text="Reference Partitions:").grid(row=0, column=0, sticky='w', pady=2)
        self.partitions_var = tk.IntVar(value=12)
        part_spin = ttk.Spinbox(self.advanced_frame, from_=6, to=20, textvariable=self.partitions_var, width=10)
        part_spin.grid(row=0, column=1, sticky='w', padx=5, pady=2)

        ttk.Label(self.advanced_frame, text="Crossover Eta:").grid(row=1, column=0, sticky='w', pady=2)
        self.eta_c_var = tk.DoubleVar(value=20.0)
        eta_c_spin = ttk.Spinbox(self.advanced_frame, from_=10.0, to=30.0, increment=1.0,
                                 textvariable=self.eta_c_var, width=10)
        eta_c_spin.grid(row=1, column=1, sticky='w', padx=5, pady=2)

        ttk.Label(self.advanced_frame, text="Mutation Eta:").grid(row=2, column=0, sticky='w', pady=2)
        self.eta_m_var = tk.DoubleVar(value=20.0)
        eta_m_spin = ttk.Spinbox(self.advanced_frame, from_=10.0, to=30.0, increment=1.0,
                                 textvariable=self.eta_m_var, width=10)
        eta_m_spin.grid(row=2, column=1, sticky='w', padx=5, pady=2)

        # Bayesian Optimization settings
        self.bayesian_frame = ttk.LabelFrame(parent, text="Bayesian Optimization Settings", padding=10)
        self.bayesian_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(self.bayesian_frame, text="Initial Samples:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.initial_samples_var = tk.IntVar(value=10)
        initial_spin = ttk.Spinbox(self.bayesian_frame, from_=5, to=50, textvariable=self.initial_samples_var, width=10)
        initial_spin.grid(row=0, column=1, sticky='w', padx=5, pady=2)

        ttk.Label(self.bayesian_frame, text="BO Iterations:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.bo_iterations_var = tk.IntVar(value=20)
        bo_iter_spin = ttk.Spinbox(self.bayesian_frame, from_=10, to=100, textvariable=self.bo_iterations_var, width=10)
        bo_iter_spin.grid(row=1, column=1, sticky='w', padx=5, pady=2)

        ttk.Label(self.bayesian_frame, text="Acquisition Function:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.acquisition_var = tk.StringVar(value="ei")
        acq_combo = ttk.Combobox(self.bayesian_frame, textvariable=self.acquisition_var,
                                 values=["ei", "pi", "ucb"], state="readonly", width=8)
        acq_combo.grid(row=2, column=1, sticky='w', padx=5, pady=2)

        # NEW: Batch Size setting
        ttk.Label(self.bayesian_frame, text="Batch Size:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.batch_size_var = tk.IntVar(value=5)
        batch_spin = ttk.Spinbox(self.bayesian_frame, from_=1, to=20, textvariable=self.batch_size_var, width=10)
        batch_spin.grid(row=3, column=1, sticky='w', padx=5, pady=2)

        # Add tooltips/descriptions
        ttk.Label(self.bayesian_frame,
                  text="EI=Expected Improvement, PI=Probability of Improvement, UCB=Upper Confidence Bound",
                  font=('Arial', 8)).grid(row=4, column=0, columnspan=2, sticky='w', padx=5, pady=2)

        ttk.Label(self.bayesian_frame,
                  text="Batch Size: Number of individuals evaluated in parallel per iteration",
                  font=('Arial', 8)).grid(row=5, column=0, columnspan=2, sticky='w', padx=5, pady=2)

        # Initially hide advanced/bayesian frames
        self.on_algorithm_change()

    def on_algorithm_change(self, event=None):
        """Handle algorithm type change to show/hide relevant settings"""
        algo_type = self.algo_var.get()

        if algo_type == "Simple":
            self.advanced_frame.pack_forget()
            self.bayesian_frame.pack_forget()
        elif algo_type == "Advanced":
            self.advanced_frame.pack(fill='x', padx=10, pady=5)
            self.bayesian_frame.pack_forget()
        elif algo_type == "Bayesian":
            self.advanced_frame.pack_forget()
            self.bayesian_frame.pack(fill='x', padx=10, pady=5)

    def create_parameter_bounds(self, parent):
        bounds_frame = ttk.LabelFrame(parent, text="Optimization Parameter Bounds", padding=10)
        bounds_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Headers
        ttk.Label(bounds_frame, text="Parameter", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, padx=5, pady=5)
        ttk.Label(bounds_frame, text="Minimum", font=('Arial', 10, 'bold')).grid(
            row=0, column=1, padx=5, pady=5)
        ttk.Label(bounds_frame, text="Maximum", font=('Arial', 10, 'bold')).grid(
            row=0, column=2, padx=5, pady=5)
        ttk.Label(bounds_frame, text="Description", font=('Arial', 10, 'bold')).grid(
            row=0, column=3, padx=5, pady=5)

        # Parameter descriptions
        descriptions = {
            'dK': 'Cylinder diameter (mm)',
            'dZ': 'Piston diameter (mm)',
            'lK': 'Cylinder length (mm)',
            'lF': 'Piston length (mm)',
            'zeta': 'Discretization parameter'
        }

        # Create input widgets for each parameter
        self.bound_vars = {}
        row = 1
        for param in ['dK', 'dZ', 'lK', 'lF', 'zeta']:
            ttk.Label(bounds_frame, text=param).grid(row=row, column=0, padx=5, pady=2, sticky='w')

            # Min value
            min_var = tk.DoubleVar() if param != 'zeta' else tk.IntVar()
            min_entry = ttk.Entry(bounds_frame, textvariable=min_var, width=10)
            min_entry.grid(row=row, column=1, padx=5, pady=2)

            # Max value
            max_var = tk.DoubleVar() if param != 'zeta' else tk.IntVar()
            max_entry = ttk.Entry(bounds_frame, textvariable=max_var, width=10)
            max_entry.grid(row=row, column=2, padx=5, pady=2)

            # Description
            ttk.Label(bounds_frame, text=descriptions[param]).grid(
                row=row, column=3, padx=5, pady=2, sticky='w')

            self.bound_vars[param] = {'min': min_var, 'max': max_var}
            row += 1

        # Preset buttons
        preset_frame = ttk.Frame(bounds_frame)
        preset_frame.grid(row=row, column=0, columnspan=4, pady=10)

        ttk.Button(preset_frame, text="Load Default",
                   command=self.load_default_bounds).pack(side='left', padx=5)
        ttk.Button(preset_frame, text="Load Conservative",
                   command=self.load_conservative_bounds).pack(side='left', padx=5)
        ttk.Button(preset_frame, text="Load Aggressive",
                   command=self.load_aggressive_bounds).pack(side='left', padx=5)

    def create_fixed_parameters(self, parent):
        """Create fixed parameters tab for LZ0 and LKG calculations"""

        # Main frame with description
        main_frame = ttk.LabelFrame(parent, text="Fixed Parameters for Calculations", padding=10)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Description
        desc_text = """These fixed parameters are used for calculating LZ0 and LKG values during optimization:
• LZ0 = LZ - lF (calculated for each iteration)
• LKG = Longest Gap Length - (Max lK - current lK) (calculated for each iteration)"""

        desc_label = ttk.Label(main_frame, text=desc_text, justify='left', foreground='blue')
        desc_label.pack(anchor='w', pady=(0, 10))

        # LZ0 calculation parameters
        lz_frame = ttk.LabelFrame(main_frame, text="LZ0 Calculation Parameters", padding=10)
        lz_frame.pack(fill='x', pady=5)

        ttk.Label(lz_frame, text="LZ (Fixed Base Value):").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.lz_var = tk.DoubleVar(value=21.358)
        lz_entry = ttk.Entry(lz_frame, textvariable=self.lz_var, width=12)
        lz_entry.grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(lz_frame, text="mm").grid(row=0, column=2, sticky='w', padx=2, pady=2)
        ttk.Label(lz_frame, text="Base value for LZ0 calculation").grid(row=0, column=3, sticky='w', padx=10, pady=2)

        # Show calculation formula
        formula_label = ttk.Label(lz_frame, text="Formula: LZ0 = LZ - lF", font=('Arial', 9, 'italic'))
        formula_label.grid(row=1, column=0, columnspan=4, sticky='w', padx=5, pady=5)

        # LKG calculation parameters
        lkg_frame = ttk.LabelFrame(main_frame, text="LKG Calculation Parameters", padding=10)
        lkg_frame.pack(fill='x', pady=5)

        ttk.Label(lkg_frame, text="Longest Gap Length:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.longest_gap_var = tk.DoubleVar(value=51.62)
        gap_entry = ttk.Entry(lkg_frame, textvariable=self.longest_gap_var, width=12)
        gap_entry.grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(lkg_frame, text="mm").grid(row=0, column=2, sticky='w', padx=2, pady=2)
        ttk.Label(lkg_frame, text="Fixed gap length value").grid(row=0, column=3, sticky='w', padx=10, pady=2)

        ttk.Label(lkg_frame, text="Max lK:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.max_lk_var = tk.DoubleVar(value=70.0)
        max_lk_entry = ttk.Entry(lkg_frame, textvariable=self.max_lk_var, width=12)
        max_lk_entry.grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(lkg_frame, text="mm").grid(row=1, column=2, sticky='w', padx=2, pady=2)
        ttk.Label(lkg_frame, text="Maximum lK value from bounds").grid(row=1, column=3, sticky='w', padx=10, pady=2)

        # Show calculation formula
        formula_label2 = ttk.Label(lkg_frame, text="Formula: LKG = Longest Gap Length - (Max lK - current lK)",
                                   font=('Arial', 9, 'italic'))
        formula_label2.grid(row=2, column=0, columnspan=4, sticky='w', padx=5, pady=5)

        # Calculation example
        example_frame = ttk.LabelFrame(main_frame, text="Calculation Example", padding=10)
        example_frame.pack(fill='x', pady=5)

        self.example_text = tk.Text(example_frame, height=6, width=80, font=('Courier', 9))
        self.example_text.pack(fill='x', pady=5)

        # Update example button
        ttk.Button(example_frame, text="Update Example",
                   command=self.update_calculation_example).pack(pady=5)

        # Load initial example
        self.update_calculation_example()

        # Auto-sync button
        sync_frame = ttk.Frame(main_frame)
        sync_frame.pack(fill='x', pady=10)

        ttk.Button(sync_frame, text="Auto-sync Max lK from Parameter Bounds",
                   command=self.sync_max_lk).pack(side='left', padx=5)

        ttk.Label(sync_frame, text="(Updates Max lK to match the maximum lK bound)",
                  font=('Arial', 8, 'italic')).pack(side='left', padx=10)

    def update_calculation_example(self):
        """Update the calculation example based on current values"""
        try:
            lz_val = self.lz_var.get()
            gap_length = self.longest_gap_var.get()
            max_lk = self.max_lk_var.get()

            # Use middle values from bounds for example
            if hasattr(self, 'bound_vars'):
                try:
                    lf_example = (self.bound_vars['lF']['min'].get() + self.bound_vars['lF']['max'].get()) / 2
                    lk_example = (self.bound_vars['lK']['min'].get() + self.bound_vars['lK']['max'].get()) / 2
                except:
                    lf_example = 35.0  # fallback
                    lk_example = 60.0  # fallback
            else:
                lf_example = 35.0
                lk_example = 60.0

            # Calculate examples
            lz0_calc = lz_val - lf_example
            x = max_lk - lk_example
            lkg_calc = gap_length - x

            example_text = f"""Example with current values:

Given: lF = {lf_example:.1f} mm, lK = {lk_example:.1f} mm

LZ0 Calculation:
  LZ0 = {lz_val:.3f} - {lf_example:.1f} = {lz0_calc:.3f} mm

LKG Calculation:
  x = {max_lk:.1f} - {lk_example:.1f} = {x:.1f} mm
  LKG = {gap_length:.2f} - {x:.1f} = {lkg_calc:.2f} mm"""

            self.example_text.delete('1.0', tk.END)
            self.example_text.insert('1.0', example_text)

        except Exception as e:
            self.example_text.delete('1.0', tk.END)
            self.example_text.insert('1.0', f"Error calculating example: {e}")

    def sync_max_lk(self):
        """Sync Max lK with the maximum lK bound"""
        try:
            if hasattr(self, 'bound_vars') and 'lK' in self.bound_vars:
                max_lk_bound = self.bound_vars['lK']['max'].get()
                self.max_lk_var.set(max_lk_bound)
                self.update_calculation_example()
                messagebox.showinfo("Sync Complete", f"Max lK updated to {max_lk_bound:.1f} mm")
            else:
                messagebox.showwarning("Sync Failed", "Parameter bounds not available yet")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to sync Max lK: {e}")

    def create_constraints_tab(self, parent):
        """Create constraints configuration tab"""

        # Main constraint frame
        main_frame = ttk.LabelFrame(parent, text="Parameter Constraints", padding=10)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Built-in constraints section
        builtin_frame = ttk.LabelFrame(main_frame, text="Built-in Constraints", padding=10)
        builtin_frame.pack(fill='x', pady=5)

        # Constraint 1: dK < dZ
        constraint1_frame = ttk.Frame(builtin_frame)
        constraint1_frame.pack(fill='x', pady=5)

        self.dk_dz_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(constraint1_frame, text="dK < dZ (dK must be less than dZ)",
                        variable=self.dk_dz_var).pack(anchor='w')

        # Constraint 2: dZ - dK difference range
        constraint2_frame = ttk.LabelFrame(builtin_frame, text="dZ - dK Difference Range", padding=5)
        constraint2_frame.pack(fill='x', pady=5)

        self.dz_dk_range_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(constraint2_frame, text="Enable dZ - dK difference constraint",
                        variable=self.dz_dk_range_var).pack(anchor='w')

        # dZ - dK range inputs
        range_input_frame = ttk.Frame(constraint2_frame)
        range_input_frame.pack(fill='x', pady=5)

        ttk.Label(range_input_frame, text="Minimum difference:").grid(row=0, column=0, sticky='w', padx=5)
        self.min_diff_var = tk.DoubleVar(value=0.1)
        min_diff_entry = ttk.Entry(range_input_frame, textvariable=self.min_diff_var, width=10)
        min_diff_entry.grid(row=0, column=1, padx=5)

        ttk.Label(range_input_frame, text="Maximum difference:").grid(row=0, column=2, sticky='w', padx=5)
        self.max_diff_var = tk.DoubleVar(value=0.8)
        max_diff_entry = ttk.Entry(range_input_frame, textvariable=self.max_diff_var, width=10)
        max_diff_entry.grid(row=0, column=3, padx=5)

        # Custom constraints section
        custom_frame = ttk.LabelFrame(main_frame, text="Custom Constraints", padding=10)
        custom_frame.pack(fill='both', expand=True, pady=5)

        # Instructions
        instructions = ttk.Label(custom_frame,
                                 text="Custom constraints can be added programmatically in constraints.py\n"
                                      "Examples: lK >= 1.5 * lF, geometric ratios, manufacturing limits, etc.",
                                 justify='left')
        instructions.pack(anchor='w', pady=5)

        # Constraint validation section
        validation_frame = ttk.LabelFrame(main_frame, text="Constraint Validation", padding=10)
        validation_frame.pack(fill='x', pady=5)

        # Test parameters button
        test_button_frame = ttk.Frame(validation_frame)
        test_button_frame.pack(fill='x', pady=5)

        ttk.Button(test_button_frame, text="Test Current Parameters",
                   command=self.test_constraints).pack(side='left', padx=5)
        ttk.Button(test_button_frame, text="Generate Valid Parameters",
                   command=self.generate_valid_params).pack(side='left', padx=5)

        # Results display
        self.constraint_results = tk.Text(validation_frame, height=8, width=80)
        self.constraint_results.pack(fill='both', expand=True, pady=5)

        # Scrollbar for results
        scrollbar = ttk.Scrollbar(validation_frame, orient="vertical", command=self.constraint_results.yview)
        scrollbar.pack(side="right", fill="y")
        self.constraint_results.configure(yscrollcommand=scrollbar.set)

    def create_system_info(self, parent):
        info_frame = ttk.LabelFrame(parent, text="System Information", padding=10)
        info_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # CPU Information
        cpu_frame = ttk.LabelFrame(info_frame, text="CPU Configuration", padding=10)
        cpu_frame.pack(fill='x', pady=5)

        total_cores = psutil.cpu_count()
        available_cores = list(range(1, total_cores))

        info_text = f"""
Total CPU Cores: {total_cores}
Core 0: RESERVED for system
Available Cores: {available_cores}
Cores for Optimization: {len(available_cores)}

Each fsti_gap.exe will use cores 1-{total_cores - 1}
Parallel Workers: {len(available_cores)}
        """

        cpu_label = ttk.Label(cpu_frame, text=info_text.strip(), justify='left')
        cpu_label.pack(anchor='w')

        # Memory Information
        mem_frame = ttk.LabelFrame(info_frame, text="Memory Information", padding=10)
        mem_frame.pack(fill='x', pady=5)

        memory = psutil.virtual_memory()
        mem_gb = memory.total / (1024 ** 3)
        mem_available_gb = memory.available / (1024 ** 3)

        mem_text = f"""
Total Memory: {mem_gb:.1f} GB
Available Memory: {mem_available_gb:.1f} GB
Memory Usage: {memory.percent:.1f}%
        """

        mem_label = ttk.Label(mem_frame, text=mem_text.strip(), justify='left')
        mem_label.pack(anchor='w')

        # Simulation Estimates
        est_frame = ttk.LabelFrame(info_frame, text="Performance Estimates", padding=10)
        est_frame.pack(fill='x', pady=5)

        # Create text widget for estimates that updates with parameter changes
        self.est_text = tk.Text(est_frame, height=8, width=60)
        self.est_text.pack()

        # Update estimates button
        ttk.Button(est_frame, text="Update Estimates",
                   command=self.update_estimates).pack(pady=5)

        self.update_estimates()

    def create_control_buttons(self):
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill='x', padx=10, pady=10)

        # Configuration management
        config_frame = ttk.LabelFrame(button_frame, text="Configuration", padding=5)
        config_frame.pack(side='left', fill='x', expand=True, padx=5)

        ttk.Button(config_frame, text="Save Config",
                   command=self.save_config).pack(side='left', padx=2)
        ttk.Button(config_frame, text="Load Config",
                   command=self.load_config).pack(side='left', padx=2)
        ttk.Button(config_frame, text="Reset to Default",
                   command=self.reset_config).pack(side='left', padx=2)

        # Optimization control
        control_frame = ttk.LabelFrame(button_frame, text="Optimization", padding=5)
        control_frame.pack(side='right', padx=5)

        ttk.Button(control_frame, text="Validate Settings",
                   command=self.validate_settings).pack(side='left', padx=2)

        # Start optimization button that runs optimization while GUI is open
        self.start_button = ttk.Button(control_frame, text="Start Optimization",
                                       command=self.start_optimization,
                                       style='Accent.TButton')
        self.start_button.pack(side='left', padx=2)

    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select Base Folder")
        if folder:
            self.folder_var.set(folder)

    def load_default_bounds(self):
        defaults = {
            'dK': {'min': 19.0, 'max': 20.0},
            'dZ': {'min': 19.2, 'max': 20.0},
            'lK': {'min': 50.0, 'max': 70.0},
            'lF': {'min': 30.0, 'max': 40.0},
            'zeta': {'min': 3, 'max': 7}
        }
        self.set_bounds(defaults)

    def load_conservative_bounds(self):
        conservative = {
            'dK': {'min': 19.3, 'max': 19.7},
            'dZ': {'min': 19.4, 'max': 19.8},
            'lK': {'min': 55.0, 'max': 65.0},
            'lF': {'min': 32.0, 'max': 38.0},
            'zeta': {'min': 4, 'max': 6}
        }
        self.set_bounds(conservative)

    def load_aggressive_bounds(self):
        aggressive = {
            'dK': {'min': 18.5, 'max': 20.5},
            'dZ': {'min': 18.8, 'max': 20.2},
            'lK': {'min': 45.0, 'max': 75.0},
            'lF': {'min': 25.0, 'max': 45.0},
            'zeta': {'min': 2, 'max': 8}
        }
        self.set_bounds(aggressive)

    def set_bounds(self, bounds_dict):
        for param, bounds in bounds_dict.items():
            self.bound_vars[param]['min'].set(bounds['min'])
            self.bound_vars[param]['max'].set(bounds['max'])
        # Update calculation example when bounds change
        self.update_calculation_example()

    def test_constraints(self):
        """Test current parameter bounds against constraints"""
        try:
            # Update constraint manager configuration
            self.update_constraint_manager()

            # Get current parameter bounds
            param_bounds = {}
            for param, vars_dict in self.bound_vars.items():
                param_bounds[param] = {
                    'min': vars_dict['min'].get(),
                    'max': vars_dict['max'].get()
                }

            # Test with middle values
            test_params = {
                'dK': (param_bounds['dK']['min'] + param_bounds['dK']['max']) / 2,
                'dZ': (param_bounds['dZ']['min'] + param_bounds['dZ']['max']) / 2,
                'lK': (param_bounds['lK']['min'] + param_bounds['lK']['max']) / 2,
                'lF': (param_bounds['lF']['min'] + param_bounds['lF']['max']) / 2,
                'zeta': int((param_bounds['zeta']['min'] + param_bounds['zeta']['max']) / 2)
            }

            # Display test parameters
            result_text = "CONSTRAINT VALIDATION TEST\n"
            result_text += "=" * 50 + "\n\n"
            result_text += "Test Parameters (middle values):\n"
            for param, value in test_params.items():
                if param == 'zeta':
                    result_text += f"  {param}: {value}\n"
                else:
                    result_text += f"  {param}: {value:.3f}\n"

            # Check constraints
            is_valid = self.constraint_manager.validate_parameters(**test_params)
            result_text += f"\nOverall Validation: {'✓ PASS' if is_valid else '✗ FAIL'}\n"

            if not is_valid:
                violated = self.constraint_manager.get_violated_constraints(**test_params)
                result_text += "\nViolated Constraints:\n"
                for v in violated:
                    result_text += f"  ✗ {v['name']}: {v['description']}\n"

                # Try to repair
                repaired = self.constraint_manager.repair_parameters(test_params, param_bounds)
                result_text += "\nRepaired Parameters:\n"
                for param, value in repaired.items():
                    if param == 'zeta':
                        result_text += f"  {param}: {value}\n"
                    else:
                        result_text += f"  {param}: {value:.3f}\n"

                # Test repaired parameters
                is_repaired_valid = self.constraint_manager.validate_parameters(**repaired)
                result_text += f"\nRepaired Validation: {'✓ PASS' if is_repaired_valid else '✗ FAIL'}\n"

            # Show constraint summary
            result_text += "\nActive Constraints:\n"
            summary = self.constraint_manager.get_constraints_summary()
            for constraint in summary:
                status = "✓" if constraint['active'] else "✗"
                result_text += f"  {status} {constraint['name']}: {constraint['description']}\n"
                if constraint['config']:
                    result_text += f"      Config: {constraint['config']}\n"

            # Display results
            self.constraint_results.delete('1.0', tk.END)
            self.constraint_results.insert('1.0', result_text)

        except Exception as e:
            error_text = f"Error testing constraints: {e}"
            self.constraint_results.delete('1.0', tk.END)
            self.constraint_results.insert('1.0', error_text)

    def generate_valid_params(self):
        """Generate valid parameters that satisfy all constraints"""
        try:
            # Update constraint manager configuration
            self.update_constraint_manager()

            # Get current parameter bounds
            param_bounds = {}
            for param, vars_dict in self.bound_vars.items():
                param_bounds[param] = {
                    'min': vars_dict['min'].get(),
                    'max': vars_dict['max'].get()
                }

            result_text = "VALID PARAMETER GENERATION\n"
            result_text += "=" * 50 + "\n\n"

            # Generate multiple valid parameter sets
            for i in range(5):
                valid_params = self.constraint_manager.generate_valid_parameters(param_bounds)

                if valid_params:
                    result_text += f"Valid Set {i + 1}:\n"
                    param_names = ['dK', 'dZ', 'lK', 'lF', 'zeta']
                    for j, param in enumerate(param_names):
                        if param == 'zeta':
                            result_text += f"  {param}: {int(valid_params[j])}\n"
                        else:
                            result_text += f"  {param}: {valid_params[j]:.3f}\n"

                    # Verify
                    param_dict = {param_names[j]: valid_params[j] for j in range(len(param_names))}
                    is_valid = self.constraint_manager.validate_parameters(**param_dict)
                    result_text += f"  Validation: {'✓ PASS' if is_valid else '✗ FAIL'}\n\n"
                else:
                    result_text += f"Set {i + 1}: ✗ Could not generate valid parameters\n\n"

            # Display results
            self.constraint_results.delete('1.0', tk.END)
            self.constraint_results.insert('1.0', result_text)

        except Exception as e:
            error_text = f"Error generating parameters: {e}"
            self.constraint_results.delete('1.0', tk.END)
            self.constraint_results.insert('1.0', error_text)

    def update_constraint_manager(self):
        """Update constraint manager with current GUI settings"""
        # Update dK < dZ constraint
        if self.dk_dz_var.get():
            self.constraint_manager.activate_constraint("dK_less_than_dZ")
        else:
            self.constraint_manager.deactivate_constraint("dK_less_than_dZ")

        # Update dZ - dK difference constraint
        if self.dz_dk_range_var.get():
            self.constraint_manager.activate_constraint("dZ_dK_difference_range")
            config = {
                'min_difference': self.min_diff_var.get(),
                'max_difference': self.max_diff_var.get()
            }
            self.constraint_manager.set_constraint_config("dZ_dK_difference_range", config)
        else:
            self.constraint_manager.deactivate_constraint("dZ_dK_difference_range")

    def update_estimates(self):
        try:
            algo_type = self.algo_var.get()
            total_cores = psutil.cpu_count()
            available_cores = total_cores - 1

            if algo_type == "Bayesian":
                # Bayesian Optimization estimates with batch support
                initial_samples = self.initial_samples_var.get()
                bo_iterations = self.bo_iterations_var.get()
                batch_size = self.batch_size_var.get()

                total_evaluations = initial_samples + (bo_iterations * batch_size)

                # Time estimates - initial batch + subsequent batches
                initial_time = (initial_samples * 45) / available_cores
                iteration_time = (bo_iterations * batch_size * 45) / available_cores

                total_time_seconds = initial_time + iteration_time
                total_hours = total_time_seconds / 3600

                estimate_text = f"""Bayesian Optimization Estimates (BATCH PARALLEL):

Initial Samples: {initial_samples} (parallel batch)
BO Iterations: {bo_iterations}
Batch Size: {batch_size} individuals per iteration
Total Evaluations: {total_evaluations}

Parallel Workers: {available_cores}
Acquisition Function: {self.acquisition_var.get().upper()}

Estimated Time per Evaluation: 30-60 seconds
Initial Batch Time: {initial_time / 3600:.1f} hours
Iteration Batches Time: {iteration_time / 3600:.1f} hours
Total Estimated Time: {total_hours:.1f} hours

Note: BO with batch evaluation combines the efficiency
of BO (smart point selection) with parallel evaluation
like NSGA-III populations.
                """
            else:
                # NSGA-III estimates (original logic)
                pop_size = self.pop_var.get()
                generations = self.gen_var.get()
                total_evaluations = pop_size * generations

                min_time_per_eval = 30
                max_time_per_eval = 60
                parallel_evaluations = max(1, available_cores)

                min_total_time = (total_evaluations * min_time_per_eval) / parallel_evaluations
                max_total_time = (total_evaluations * max_time_per_eval) / parallel_evaluations

                min_hours = min_total_time / 3600
                max_hours = max_total_time / 3600

                estimate_text = f"""NSGA-III Performance Estimates:

Population Size: {pop_size}
Generations: {generations}
Total Evaluations: {total_evaluations}

Parallel Workers: {available_cores}
Concurrent Simulations: {available_cores}

Estimated Time per Evaluation: {min_time_per_eval}-{max_time_per_eval} seconds
Estimated Total Time: {min_hours:.1f} - {max_hours:.1f} hours

Note: Actual times depend on simulation complexity,
hardware performance, and system load.
                """

            self.est_text.delete('1.0', tk.END)
            self.est_text.insert('1.0', estimate_text)

        except Exception as e:
            self.est_text.delete('1.0', tk.END)
            self.est_text.insert('1.0', f"Error calculating estimates: {e}")

    def get_current_config(self):
        """Get current configuration from GUI"""
        config = {}

        # Basic settings
        config['base_folder'] = self.folder_var.get()
        config['algorithm_type'] = self.algo_var.get()

        # Algorithm-specific settings
        if config['algorithm_type'] == 'Bayesian':
            config['initial_samples'] = self.initial_samples_var.get()
            config['bo_iterations'] = self.bo_iterations_var.get()
            config['acquisition_function'] = self.acquisition_var.get()
            config['batch_size'] = self.batch_size_var.get()  # NEW: Add batch size
            # For compatibility, set population_size and generations
            config['population_size'] = self.initial_samples_var.get()
            config['generations'] = 1  # Not used in BO
        else:
            config['population_size'] = self.pop_var.get()
            config['generations'] = self.gen_var.get()

        # Advanced settings (for Advanced NSGA-III)
        config['reference_partitions'] = self.partitions_var.get()
        config['eta_c'] = self.eta_c_var.get()
        config['eta_m'] = self.eta_m_var.get()

        # Parameter bounds
        config['param_bounds'] = {}
        for param, vars_dict in self.bound_vars.items():
            config['param_bounds'][param] = {
                'min': vars_dict['min'].get(),
                'max': vars_dict['max'].get()
            }

        # Fixed parameters for calculations
        config['fixed_params'] = {
            'LZ': self.lz_var.get(),
            'longest_gap_length': self.longest_gap_var.get(),
            'max_lK': self.max_lk_var.get()
        }

        # Constraints
        config['constraints'] = {
            'dK_less_than_dZ': {
                'active': self.dk_dz_var.get()
            },
            'dZ_dK_difference_range': {
                'active': self.dz_dk_range_var.get(),
                'min_difference': self.min_diff_var.get(),
                'max_difference': self.max_diff_var.get()
            }
        }

        return config

    def set_config(self, config):
        """Set GUI from configuration"""
        try:
            # Basic settings
            self.folder_var.set(config.get('base_folder', ''))
            self.algo_var.set(config.get('algorithm_type', 'Simple'))

            # Algorithm-specific settings
            if config.get('algorithm_type') == 'Bayesian':
                self.initial_samples_var.set(config.get('initial_samples', 10))
                self.bo_iterations_var.set(config.get('bo_iterations', 20))
                self.acquisition_var.set(config.get('acquisition_function', 'ei'))
                self.batch_size_var.set(config.get('batch_size', 5))  # NEW: Set batch size
            else:
                self.pop_var.set(config.get('population_size', 20))
                self.gen_var.set(config.get('generations', 10))

            # Advanced settings
            self.partitions_var.set(config.get('reference_partitions', 12))
            self.eta_c_var.set(config.get('eta_c', 20.0))
            self.eta_m_var.set(config.get('eta_m', 20.0))

            # Parameter bounds
            param_bounds = config.get('param_bounds', {})
            for param, vars_dict in self.bound_vars.items():
                if param in param_bounds:
                    vars_dict['min'].set(param_bounds[param]['min'])
                    vars_dict['max'].set(param_bounds[param]['max'])

            # Fixed parameters
            fixed_params = config.get('fixed_params', {})
            self.lz_var.set(fixed_params.get('LZ', 21.358))
            self.longest_gap_var.set(fixed_params.get('longest_gap_length', 51.62))
            self.max_lk_var.set(fixed_params.get('max_lK', 70.0))

            # Update calculation example
            self.update_calculation_example()

            # Constraints
            constraints = config.get('constraints', {})

            # dK < dZ constraint
            dk_dz_config = constraints.get('dK_less_than_dZ', {})
            self.dk_dz_var.set(dk_dz_config.get('active', True))

            # dZ - dK difference constraint
            dz_dk_config = constraints.get('dZ_dK_difference_range', {})
            self.dz_dk_range_var.set(dz_dk_config.get('active', True))
            self.min_diff_var.set(dz_dk_config.get('min_difference', 0.1))
            self.max_diff_var.set(dz_dk_config.get('max_difference', 0.8))

            # Update GUI visibility based on algorithm
            self.on_algorithm_change()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {e}")

    def save_config(self):
        """Save current configuration to file"""
        try:
            config = self.get_current_config()
            filename = filedialog.asksaveasfilename(
                title="Save Configuration",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if filename:
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                messagebox.showinfo("Success", f"Configuration saved to {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def load_config(self):
        """Load configuration from file"""
        try:
            filename = filedialog.askopenfilename(
                title="Load Configuration",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if filename:
                with open(filename, 'r') as f:
                    config = json.load(f)

                self.set_config(config)
                messagebox.showinfo("Success", f"Configuration loaded from {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {e}")

    def load_default_config(self):
        """Load default configuration"""
        default_config = {
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
            'batch_size': 5,  # NEW: Default batch size
            'param_bounds': {
                'dK': {'min': 19.0, 'max': 20.0},
                'dZ': {'min': 19.2, 'max': 20.0},
                'lK': {'min': 50.0, 'max': 70.0},
                'lF': {'min': 30.0, 'max': 40.0},
                'zeta': {'min': 3, 'max': 7}
            },
            'fixed_params': {
                'LZ': 21.358,
                'longest_gap_length': 51.62,
                'max_lK': 70.0
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
        self.set_config(default_config)

    def reset_config(self):
        """Reset to default configuration"""
        if messagebox.askyesno("Reset Configuration",
                               "Are you sure you want to reset all settings to default?"):
            self.load_default_config()

    def validate_settings(self):
        """Validate current settings"""
        try:
            config = self.get_current_config()
            errors = []

            # Check base folder
            if not config['base_folder']:
                errors.append("Base folder is required")
            elif not os.path.exists(config['base_folder']):
                errors.append("Base folder does not exist")
            else:
                # Check for required files
                base_path = Path(config['base_folder'])
                if not (base_path / 'input').exists():
                    errors.append("Input folder not found in base folder")
                if not (base_path / 'fsti_gap.exe').exists():
                    errors.append("fsti_gap.exe not found in base folder")

            # Check parameter bounds
            for param, bounds in config['param_bounds'].items():
                if bounds['min'] >= bounds['max']:
                    errors.append(f"Invalid bounds for {param}: min must be less than max")

            # Check fixed parameters
            fixed_params = config.get('fixed_params', {})
            if fixed_params.get('LZ', 0) <= 0:
                errors.append("LZ must be positive")
            if fixed_params.get('longest_gap_length', 0) <= 0:
                errors.append("Longest Gap Length must be positive")
            if fixed_params.get('max_lK', 0) <= 0:
                errors.append("Max lK must be positive")

            # Validate Max lK against lK bounds
            max_lk = fixed_params.get('max_lK', 0)
            lk_max_bound = config['param_bounds']['lK']['max']
            if max_lk < lk_max_bound:
                errors.append(f"Max lK ({max_lk}) should be >= lK maximum bound ({lk_max_bound})")

            # Algorithm-specific validation
            if config['algorithm_type'] == 'Bayesian':
                if config.get('initial_samples', 0) < 3:
                    errors.append("Initial samples must be at least 3 for Bayesian Optimization")
                if config.get('bo_iterations', 0) < 1:
                    errors.append("BO iterations must be at least 1")
                if config.get('batch_size', 0) < 1:
                    errors.append("Batch size must be at least 1")
            else:
                if config.get('population_size', 0) < 4:
                    errors.append("Population size must be at least 4")
                if config.get('generations', 0) < 1:
                    errors.append("Generations must be at least 1")

            # Check constraints
            constraints = config.get('constraints', {})
            dz_dk_config = constraints.get('dZ_dK_difference_range', {})
            if dz_dk_config.get('active', False):
                min_diff = dz_dk_config.get('min_difference', 0.1)
                max_diff = dz_dk_config.get('max_difference', 0.8)
                if min_diff >= max_diff:
                    errors.append("dZ-dK minimum difference must be less than maximum difference")

            if errors:
                messagebox.showerror("Validation Errors", "\n".join(errors))
                return False
            else:
                messagebox.showinfo("Validation", "All settings are valid!")
                return True

        except Exception as e:
            messagebox.showerror("Validation Error", f"Error during validation: {e}")
            return False

    def start_optimization(self):
        """Start the optimization process with progress tracking - runs while GUI is open"""
        if self.validate_settings():
            config = self.get_current_config()

            # Algorithm-specific confirmation message
            fixed_params = config.get('fixed_params', {})
            fixed_params_info = f"""
Fixed Parameters for Calculations:
- LZ: {fixed_params.get('LZ', 'Not set')} mm
- Longest Gap Length: {fixed_params.get('longest_gap_length', 'Not set')} mm
- Max lK: {fixed_params.get('max_lK', 'Not set')} mm

During optimization:
- LZ0 will be calculated as: LZ - lF
- LKG will be calculated as: Longest Gap Length - (Max lK - current lK)"""

            if config['algorithm_type'] == 'Bayesian':
                batch_size = config.get('batch_size', 5)
                total_evals = config.get('initial_samples', 10) + (config.get('bo_iterations', 20) * batch_size)

                message = f"""Start Bayesian Optimization with the following settings?

Base Folder: {config['base_folder']}
Algorithm: Bayesian Optimization (BATCH PARALLEL)
Initial Samples: {config.get('initial_samples', 10)}
BO Iterations: {config.get('bo_iterations', 20)}
Batch Size: {batch_size} individuals per iteration
Acquisition Function: {config.get('acquisition_function', 'EI').upper()}
Total Evaluations: ~{total_evals}
{fixed_params_info}

Constraints:
- dK < dZ: {'Enabled' if config['constraints']['dK_less_than_dZ']['active'] else 'Disabled'}
- dZ-dK range: {'Enabled' if config['constraints']['dZ_dK_difference_range']['active'] else 'Disabled'}

BATCH PARALLEL MODE: Like NSGA-III, {batch_size} individuals 
will be evaluated simultaneously per iteration!
This may take several hours to complete.

The optimization will run while the GUI remains open.
                """
            else:
                message = f"""Start optimization with the following settings?

Base Folder: {config['base_folder']}
Algorithm: {config['algorithm_type']} NSGA-III
Population: {config.get('population_size', 20)}
Generations: {config.get('generations', 10)}
{fixed_params_info}

Constraints:
- dK < dZ: {'Enabled' if config['constraints']['dK_less_than_dZ']['active'] else 'Disabled'}
- dZ-dK range: {'Enabled' if config['constraints']['dZ_dK_difference_range']['active'] else 'Disabled'}

This may take several hours to complete.

The optimization will run while the GUI remains open.
                """

            if messagebox.askyesno("Start Optimization", message):
                # Store config and start optimization in GUI
                self.config = config
                self.optimization_started = True

                # Disable start button to prevent multiple starts
                self.start_button.config(state='disabled', text='Optimization Running...')

                # Show progress window and start optimization
                self.show_progress_window()
                self.start_optimization_thread()

    def show_progress_window(self):
        """Show optimization progress window"""
        # Create progress window
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Optimization Progress")
        self.progress_window.geometry("800x600")
        self.progress_window.transient(self.root)

        # Make it stay on top
        self.progress_window.attributes('-topmost', True)

        # Progress info frame
        info_frame = ttk.LabelFrame(self.progress_window, text="Optimization Status", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)

        # Status labels
        self.status_label = ttk.Label(info_frame, text="Starting optimization...", font=('Arial', 12, 'bold'))
        self.status_label.pack(anchor='w', pady=2)

        self.progress_label = ttk.Label(info_frame, text="Preparing...", font=('Arial', 10))
        self.progress_label.pack(anchor='w', pady=2)

        self.time_label = ttk.Label(info_frame, text="Elapsed time: 00:00:00", font=('Arial', 10))
        self.time_label.pack(anchor='w', pady=2)

        # Progress bar
        self.progress_bar = ttk.Progressbar(info_frame, mode='determinate')
        self.progress_bar.pack(fill='x', pady=5)

        # Current evaluation details
        eval_frame = ttk.LabelFrame(self.progress_window, text="Current Evaluation", padding=10)
        eval_frame.pack(fill='x', padx=10, pady=5)

        self.current_eval_label = ttk.Label(eval_frame, text="No evaluation running", font=('Arial', 10))
        self.current_eval_label.pack(anchor='w', pady=2)

        # Results display
        results_frame = ttk.LabelFrame(self.progress_window, text="Live Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.results_text = tk.Text(results_frame, height=15, width=80)
        self.results_text.pack(fill='both', expand=True, pady=5)

        # Scrollbar for results
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.results_text.configure(yscrollcommand=scrollbar.set)

        # Control buttons
        button_frame = ttk.Frame(self.progress_window)
        button_frame.pack(fill='x', padx=10, pady=5)

        self.stop_button = ttk.Button(button_frame, text="Stop Optimization",
                                      command=self.stop_optimization, state='normal')
        self.stop_button.pack(side='left', padx=5)

        self.close_button = ttk.Button(button_frame, text="Close Window",
                                       command=self.close_progress_window, state='normal')
        self.close_button.pack(side='right', padx=5)

        # Initialize progress tracking
        self.optimization_running = True
        self.start_time = time.time()
        self.update_timer()

    def start_optimization_thread(self):
        """Start optimization in a separate thread"""
        import threading
        self.optimization_thread = threading.Thread(target=self.run_optimization_with_progress)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()

    def run_optimization_with_progress(self):
        """Run optimization with progress updates"""
        try:
            # Import here to avoid circular imports
            from piston_optimizer import PistonOptimizer
            from nsga3_algorithm import SimpleNSGA3, AdvancedNSGA3
            from bayesian_optimization import BayesianOptimization
            from cpu_affinity import get_available_cores, force_exclude_core0

            # Update status
            self.update_status("Initializing optimization...")

            # Force exclude core 0 from main process
            force_exclude_core0()

            # Get available cores
            available_cores = get_available_cores()

            # Convert parameter bounds to expected format
            param_bounds = {}
            for param, bounds in self.config['param_bounds'].items():
                param_bounds[param] = (bounds['min'], bounds['max'])

            # Create optimizer with progress callback
            optimizer = PistonOptimizer(self.config['base_folder'], progress_callback=self)

            # Choose algorithm
            if self.config.get('algorithm_type', 'Simple').lower() == 'bayesian':
                optimizer_instance = BayesianOptimization(
                    optimizer=optimizer,
                    param_bounds=param_bounds,
                    available_cores=available_cores,
                    n_initial=self.config.get('initial_samples', 10),
                    n_iterations=self.config.get('bo_iterations', 20),
                    acquisition=self.config.get('acquisition_function', 'ei'),
                    constraints_config=self.config.get('constraints', {}),
                    progress_callback=self,
                    fixed_params=self.config.get('fixed_params', {}),
                    batch_size=self.config.get('batch_size', 5)  # NEW: Pass batch size
                )
            elif self.config.get('algorithm_type', 'Simple').lower() == 'advanced':
                optimizer_instance = AdvancedNSGA3(
                    optimizer=optimizer,
                    param_bounds=param_bounds,
                    available_cores=available_cores,
                    pop_size=self.config['population_size'],
                    generations=self.config['generations'],
                    n_partitions=self.config.get('reference_partitions', 12),
                    constraints_config=self.config.get('constraints', {}),
                    progress_callback=self,
                    fixed_params=self.config.get('fixed_params', {})
                )
            else:
                optimizer_instance = SimpleNSGA3(
                    optimizer=optimizer,
                    param_bounds=param_bounds,
                    available_cores=available_cores,
                    pop_size=self.config['population_size'],
                    generations=self.config['generations'],
                    constraints_config=self.config.get('constraints', {}),
                    progress_callback=self,
                    fixed_params=self.config.get('fixed_params', {})
                )

            # Run optimization
            self.update_status("Running optimization...")
            best_individuals, best_objectives = optimizer_instance.run_optimization()

            # Optimization completed
            if best_individuals:
                self.update_status("Optimization completed successfully!")
                self.update_results(f"\n✓ Found {len(best_individuals)} Pareto optimal solutions")

                # Save results
                optimizer.save_results(best_individuals, best_objectives, algorithm_type=self.config['algorithm_type'])
                report_file = optimizer.generate_summary_report(best_individuals, best_objectives, self.config)

                self.update_results(f"✓ Results saved to optimization folder")
                self.update_results(f"✓ Summary report: {report_file}")
            else:
                self.update_status("Optimization completed - No valid solutions found")
                self.update_results("\n✗ No valid solutions found!")

        except Exception as e:
            self.update_status(f"Optimization failed: {str(e)}")
            self.update_results(f"\n✗ Error: {str(e)}")

        finally:
            self.optimization_running = False
            self.root.after(0, self.optimization_finished)

    def update_status(self, status_text):
        """Update status label (thread-safe)"""

        def update():
            if hasattr(self, 'status_label'):
                self.status_label.config(text=status_text)

        self.root.after(0, update)

    def update_progress(self, current, total, generation_or_iteration=None):
        """Update progress bar (thread-safe)"""

        def update():
            if hasattr(self, 'progress_bar'):
                percentage = (current / total) * 100 if total > 0 else 0
                self.progress_bar['value'] = percentage

                if generation_or_iteration is not None:
                    if self.config.get('algorithm_type', '').lower() == 'bayesian':
                        if generation_or_iteration == 0:
                            phase = "Initial Sampling"
                        else:
                            phase = f"BO Iteration {generation_or_iteration}"
                    else:
                        phase = f"Generation {generation_or_iteration}"

                    self.progress_label.config(text=f"{phase}: {current}/{total} evaluations ({percentage:.1f}%)")
                else:
                    self.progress_label.config(text=f"Progress: {current}/{total} ({percentage:.1f}%)")

        self.root.after(0, update)

    def update_current_evaluation(self, individual_id, params):
        """Update current evaluation details (thread-safe)"""

        def update():
            if hasattr(self, 'current_eval_label'):
                param_str = f"dK={params[0]:.3f}, dZ={params[1]:.3f}, lK={params[2]:.1f}, lF={params[3]:.1f}, zeta={int(params[4])}"
                self.current_eval_label.config(text=f"Evaluating Individual {individual_id}: {param_str}")

        self.root.after(0, update)

    def update_results(self, result_text):
        """Update results display (thread-safe)"""

        def update():
            if hasattr(self, 'results_text'):
                self.results_text.insert(tk.END, result_text + "\n")
                self.results_text.see(tk.END)

        self.root.after(0, update)

    def update_timer(self):
        """Update elapsed time timer"""
        if self.optimization_running and hasattr(self, 'time_label'):
            elapsed = time.time() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.time_label.config(text=f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")

            # Schedule next update
            self.root.after(1000, self.update_timer)

    def optimization_finished(self):
        """Called when optimization is finished"""
        # Re-enable start button
        self.start_button.config(state='normal', text='Start Optimization')

        if hasattr(self, 'stop_button'):
            self.stop_button.config(state='disabled')

        # Final time update
        if hasattr(self, 'time_label'):
            elapsed = time.time() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.time_label.config(text=f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def stop_optimization(self):
        """Stop the optimization"""
        self.optimization_running = False
        self.update_status("Stopping optimization...")
        # Note: Actual stopping of threads/processes would need to be implemented
        # in the optimization algorithms

    def close_progress_window(self):
        """Close the progress window"""
        if hasattr(self, 'progress_window'):
            self.progress_window.destroy()

    def get_final_config(self):
        """Get the final configuration - only return if optimization was started"""
        if self.optimization_started:
            return getattr(self, 'config', None)
        return None


def run_gui():
    """Run the GUI and return the configuration only if optimization was started"""
    root = tk.Tk()
    gui = OptimizationGUI(root)

    # Keep GUI running - don't quit on optimization start
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("GUI interrupted by user")

    # Only return config if optimization was actually started
    return gui.get_final_config()


if __name__ == "__main__":
    config = run_gui()
    if config:
        print("Optimization was started through GUI")
        print("Configuration used:")
        print(json.dumps(config, indent=2))
    else:
        print("GUI closed without starting optimization")