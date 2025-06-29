import matplotlib
# Use Agg backend for environments without display
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create_pareto_plot(objectives):
    """Create a matplotlib Figure showing the Pareto front.

    Parameters
    ----------
    objectives : list of tuple
        List of (mechanical_loss, volumetric_loss) pairs.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object with the Pareto scatter plot.
    """
    mech = [o[0] for o in objectives]
    vol = [o[1] for o in objectives]

    fig, ax = plt.subplots()
    ax.scatter(mech, vol, c="blue", edgecolors="black")
    ax.set_xlabel("Mechanical Loss")
    ax.set_ylabel("Volumetric Loss")
    ax.set_title("Pareto Front")
    fig.tight_layout()
    return fig
