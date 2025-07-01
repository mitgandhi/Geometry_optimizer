"""
Optimizer Constraint System
==========================

Simple and extensible constraint system for piston optimization parameters.
Allows easy addition of custom constraints without modifying the main optimizer.

Usage:
    constraint_manager = ConstraintManager()
    constraint_manager.add_constraint("dK_less_than_dZ", lambda params: params['dK'] < params['dZ'])

    # Check if parameters satisfy all constraints
    is_valid = constraint_manager.validate_parameters(dK=19.5, dZ=19.8, LKG=60.0, lF=35.0, zeta=5)

    # Generate valid parameters
    valid_params = constraint_manager.generate_valid_parameters(param_bounds)
"""

import random
import math


class ConstraintManager:
    """Simple constraint manager for optimization parameters"""

    def __init__(self):
        self.constraints = {}
        self.constraint_config = {}
        self._setup_default_constraints()

    def _setup_default_constraints(self):
        """Setup default constraints required for piston optimization"""

        # Constraint 1: dK must be less than dZ
        self.add_constraint(
            name="dK_less_than_dZ",
            constraint_func=lambda params: params['dK'] < params['dZ'],
            description="dK must be less than dZ"
        )

        # Constraint 2: dZ - dK must be within specified range
        # Default range will be set via configuration
        self.add_constraint(
            name="dZ_dK_difference_range",
            constraint_func=self._check_dz_dk_difference,
            description="dZ - dK must be within specified range"
        )

        # Set default configuration for dZ-dK difference
        self.set_constraint_config("dZ_dK_difference_range", {
            'min_difference': 0.1,
            'max_difference': 0.8
        })

    def add_constraint(self, name, constraint_func, description=""):
        """
        Add a custom constraint

        Args:
            name (str): Unique name for the constraint
            constraint_func (callable): Function that takes params dict and returns True if valid
            description (str): Human-readable description of the constraint
        """
        self.constraints[name] = {
            'func': constraint_func,
            'description': description,
            'active': True
        }
        print(f"✓ Added constraint: {name} - {description}")

    def remove_constraint(self, name):
        """Remove a constraint by name"""
        if name in self.constraints:
            del self.constraints[name]
            print(f"✓ Removed constraint: {name}")
        else:
            print(f"⚠️ Constraint not found: {name}")

    def activate_constraint(self, name):
        """Activate a constraint"""
        if name in self.constraints:
            self.constraints[name]['active'] = True
            print(f"✓ Activated constraint: {name}")

    def deactivate_constraint(self, name):
        """Deactivate a constraint (but keep it in the system)"""
        if name in self.constraints:
            self.constraints[name]['active'] = False
            print(f"✓ Deactivated constraint: {name}")

    def set_constraint_config(self, constraint_name, config):
        """Set configuration parameters for a constraint"""
        self.constraint_config[constraint_name] = config
        print(f"✓ Set config for {constraint_name}: {config}")

    def get_constraint_config(self, constraint_name, default=None):
        """Get configuration for a constraint"""
        return self.constraint_config.get(constraint_name, default)

    def _check_dz_dk_difference(self, params):
        """Check if dZ - dK is within the specified range"""
        config = self.get_constraint_config("dZ_dK_difference_range", {})
        min_diff = config.get('min_difference', 0.1)
        max_diff = config.get('max_difference', 0.8)

        difference = params['dZ'] - params['dK']
        return min_diff <= difference <= max_diff

    def validate_parameters(self, **params):
        """
        Validate parameters against all active constraints

        Args:
            **params: Parameter values (dK, dZ, LKG, lF, zeta)

        Returns:
            bool: True if all constraints are satisfied
        """
        for name, constraint in self.constraints.items():
            if not constraint['active']:
                continue

            try:
                if not constraint['func'](params):
                    return False
            except Exception as e:
                print(f"⚠️ Error in constraint {name}: {e}")
                return False

        return True

    def get_violated_constraints(self, **params):
        """
        Get list of violated constraints

        Returns:
            list: Names and descriptions of violated constraints
        """
        violated = []

        for name, constraint in self.constraints.items():
            if not constraint['active']:
                continue

            try:
                if not constraint['func'](params):
                    violated.append({
                        'name': name,
                        'description': constraint['description']
                    })
            except Exception as e:
                violated.append({
                    'name': name,
                    'description': f"Error: {e}"
                })

        return violated

    def generate_valid_parameters(self, param_bounds, max_attempts=1000):
        """
        Generate random parameters that satisfy all constraints

        Args:
            param_bounds (dict): Parameter bounds in format {'param': {'min': x, 'max': y}} or {'param': (min, max)}
            max_attempts (int): Maximum attempts to generate valid parameters

        Returns:
            list: [dK, dZ, LKG, lF, zeta] or None if no valid parameters found
        """
        # Convert bounds format if needed (handle both dict and tuple formats)
        normalized_bounds = {}
        for param, bounds in param_bounds.items():
            if isinstance(bounds, (tuple, list)):
                # Convert (min, max) to {'min': min, 'max': max}
                normalized_bounds[param] = {'min': bounds[0], 'max': bounds[1]}
            elif isinstance(bounds, dict):
                # Already in correct format
                normalized_bounds[param] = bounds
            else:
                raise ValueError(f"Invalid bounds format for {param}: {bounds}")

        for attempt in range(max_attempts):

            params = {
                'dK': random.uniform(normalized_bounds['dK']['min'], normalized_bounds['dK']['max']),
                'dZ': random.uniform(normalized_bounds['dZ']['min'], normalized_bounds['dZ']['max']),
                'LKG': random.uniform(normalized_bounds['LKG']['min'], normalized_bounds['LKG']['max']),
                'lF': random.uniform(normalized_bounds['lF']['min'], normalized_bounds['lF']['max']),
                'zeta': random.randint(normalized_bounds['zeta']['min'], normalized_bounds['zeta']['max'])
            }

            

            # Check if parameters satisfy all constraints
            if self.validate_parameters(**params):
                return [params['dK'], params['dZ'], params['LKG'], params['lF'], params['zeta']]

        print(f"⚠️ Could not generate valid parameters after {max_attempts} attempts")
        return None

    def repair_parameters(self, params_dict, param_bounds):
        """
        Try to repair invalid parameters to make them satisfy constraints

        Args:
            params_dict (dict): Current parameter values
            param_bounds (dict): Parameter bounds in format {'param': {'min': x, 'max': y}} or {'param': (min, max)}

        Returns:
            dict: Repaired parameters or original if repair failed
        """
        # Convert bounds format if needed
        normalized_bounds = {}
        for param, bounds in param_bounds.items():
            if isinstance(bounds, (tuple, list)):
                normalized_bounds[param] = {'min': bounds[0], 'max': bounds[1]}
            elif isinstance(bounds, dict):
                normalized_bounds[param] = bounds
            else:
                raise ValueError(f"Invalid bounds format for {param}: {bounds}")

        repaired = params_dict.copy()

        # Repair dK < dZ constraint
        if repaired['dK'] >= repaired['dZ']:
            # Adjust dK to be slightly less than dZ
            repaired['dK'] = repaired['dZ'] - 0.05
            # Ensure dK is still within bounds
            repaired['dK'] = max(repaired['dK'], normalized_bounds['dK']['min'])

        # Repair dZ - dK difference constraint
        config = self.get_constraint_config("dZ_dK_difference_range", {})
        min_diff = config.get('min_difference', 0.1)
        max_diff = config.get('max_difference', 0.8)

        difference = repaired['dZ'] - repaired['dK']

        if difference < min_diff:
            # Increase dZ or decrease dK
            repaired['dZ'] = repaired['dK'] + min_diff
            # Check bounds
            if repaired['dZ'] > normalized_bounds['dZ']['max']:
                repaired['dZ'] = normalized_bounds['dZ']['max']
                repaired['dK'] = repaired['dZ'] - min_diff
                repaired['dK'] = max(repaired['dK'], normalized_bounds['dK']['min'])

        elif difference > max_diff:
            # Decrease dZ or increase dK
            repaired['dZ'] = repaired['dK'] + max_diff
            # Check bounds
            if repaired['dZ'] > normalized_bounds['dZ']['max']:
                repaired['dZ'] = normalized_bounds['dZ']['max']
                repaired['dK'] = repaired['dZ'] - max_diff
                repaired['dK'] = max(repaired['dK'], normalized_bounds['dK']['min'])

        # Ensure zeta is even and within bounds
        zmin = normalized_bounds['zeta']['min']
        zmax = normalized_bounds['zeta']['max']
        zeta_even = int(round(repaired.get('zeta', zmin) / 2.0)) * 2
        if zeta_even < zmin:
            zeta_even = zmin if zmin % 2 == 0 else zmin + 1
        if zeta_even > zmax:
            zeta_even = zmax if zmax % 2 == 0 else zmax - 1
        repaired['zeta'] = zeta_even

        return repaired

    def list_constraints(self):
        """List all constraints and their status"""
        print("\nActive Constraints:")
        print("-" * 50)

        for name, constraint in self.constraints.items():
            status = "✓ ACTIVE" if constraint['active'] else "✗ INACTIVE"
            print(f"{status}: {name}")
            if constraint['description']:
                print(f"   Description: {constraint['description']}")

            # Show configuration if available
            config = self.get_constraint_config(name)
            if config:
                print(f"   Config: {config}")
            print()

    def get_constraints_summary(self):
        """Get summary of constraints for GUI display"""
        summary = []

        for name, constraint in self.constraints.items():
            summary.append({
                'name': name,
                'description': constraint['description'],
                'active': constraint['active'],
                'config': self.get_constraint_config(name, {})
            })

        return summary


# # Example usage and testing
# def example_usage():
#     """Example of how to use the constraint system"""
#
#     # Create constraint manager
#     cm = ConstraintManager()
#
#     # Add a custom constraint
#     def lK_lF_ratio_constraint(params):
#         """lK should be at least 1.5 times lF"""
#         return params['lK'] >= 1.5 * params['lF']
#
#     cm.add_constraint(
#         name="lK_lF_ratio",
#         constraint_func=lK_lF_ratio_constraint,
#         description="lK must be at least 1.5 times lF"
#     )
#
#     # Set configuration for dZ-dK difference
#     cm.set_constraint_config("dZ_dK_difference_range", {
#         'min_difference': 0.2,
#         'max_difference': 0.6
#     })
#
#     # Example parameter bounds
#     param_bounds = {
#         'dK': {'min': 19.0, 'max': 20.0},
#         'dZ': {'min': 19.2, 'max': 20.0},
#         'lK': {'min': 50.0, 'max': 70.0},
#         'lF': {'min': 30.0, 'max': 40.0},
#         'zeta': {'min': 3, 'max': 7}
#     }
#
#     # Test parameter validation
#     test_params = {
#         'dK': 19.3,
#         'dZ': 19.7,  # dZ > dK ✓, difference = 0.4 ✓
#         'lK': 60.0,
#         'lF': 35.0,  # lK/lF = 1.71 ✓
#         'zeta': 5
#     }
#
#     print("Testing parameter validation:")
#     is_valid = cm.validate_parameters(**test_params)
#     print(f"Parameters valid: {is_valid}")
#
#     if not is_valid:
#         violated = cm.get_violated_constraints(**test_params)
#         print("Violated constraints:")
#         for v in violated:
#             print(f"  - {v['name']}: {v['description']}")
#
#     # Generate valid parameters
#     print("\nGenerating valid parameters:")
#     valid_params = cm.generate_valid_parameters(param_bounds)
#     if valid_params:
#         print(f"Generated: dK={valid_params[0]:.3f}, dZ={valid_params[1]:.3f}, "
#               f"lK={valid_params[2]:.1f}, lF={valid_params[3]:.1f}, zeta={valid_params[4]}")
#
#         # Verify they're valid
#         param_dict = {
#             'dK': valid_params[0], 'dZ': valid_params[1], 'lK': valid_params[2],
#             'lF': valid_params[3], 'zeta': valid_params[4]
#         }
#         print(f"Verification: {cm.validate_parameters(**param_dict)}")
#
#     # List all constraints
#     cm.list_constraints()
#
#
# if __name__ == "__main__":
#     example_usage()