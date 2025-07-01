import re
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PARAM_PATTERN = re.compile(
    r"dK(?P<dK>[-\d\.]+)_dZ(?P<dZ>[-\d\.]+)_lK(?P<lK>[-\d\.]+)_lF(?P<lF>[-\d\.]+)_zeta(?P<zeta>[-\d\.]+)"
)


def parse_parameters(folder_name: str) -> Dict[str, float]:
    """Extract parameter values from a simulation folder name."""
    match = PARAM_PATTERN.search(folder_name)
    if not match:
        return {}
    params = {k: float(v) for k, v in match.groupdict().items()}
    params['zeta'] = int(params['zeta'])
    return params


def parse_generation_folder(gen_folder: Path) -> List[Dict[str, float]]:
    """Parse all simulation folders in a generation/iteration folder."""
    params = []
    for child in gen_folder.iterdir():
        if child.is_dir():
            parsed = parse_parameters(child.name)
            if parsed:
                params.append(parsed)
    return params


def aggregate_mean(params_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute mean of each parameter from a list of parameter dicts."""
    if not params_list:
        return {}
    sums = {k: 0.0 for k in params_list[0]}
    for params in params_list:
        for k, v in params.items():
            sums[k] += v
    return {k: sums[k] / len(params_list) for k in sums}


def parse_algorithm_folder(base_folder: Path) -> Dict[str, List[float]]:
    """Parse an optimization folder and return mean parameter values per generation."""
    generations = {}
    for child in sorted(base_folder.iterdir()):
        if not child.is_dir():
            continue
        # Detect generation or iteration numbering
        if child.name.startswith('Generation_G'):
            idx = int(child.name.split('Generation_G')[1])
        elif child.name.startswith('Iteration_I'):
            idx = int(child.name.split('Iteration_I')[1])
        elif child.name == 'Initial_Sampling':
            idx = 0
        else:
            continue
        params_list = parse_generation_folder(child)
        mean_params = aggregate_mean(params_list)
        if mean_params:
            generations[idx] = mean_params
    # Order by generation index
    ordered = [generations[g] for g in sorted(generations)]
    result = {k: [gen[k] for gen in ordered] for k in ordered[0]}
    result['generation'] = sorted(generations)
    return result


def plot_parameter_evolution(base_folder: str, output_file: str = 'parameter_evolution.png') -> str:
    """Create a line plot of parameter means vs generation."""
    folder = Path(base_folder)
    data = parse_algorithm_folder(folder)
    generations = data.pop('generation')

    plt.figure()
    for param, values in data.items():
        plt.plot(generations, values, marker='o', label=param)
    plt.xlabel('Generation/Iteration')
    plt.ylabel('Parameter Value (mean)')
    plt.title('Parameter Evolution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    return output_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot parameter evolution from optimization folders.')
    parser.add_argument('folder', help='Path to optimization folder (e.g., simple_nsga3)')
    parser.add_argument('--output', default='parameter_evolution.png', help='Output image file path')
    args = parser.parse_args()

    outfile = plot_parameter_evolution(args.folder, args.output)
    print(f'Parameter evolution plot saved to {outfile}')
