"""IEEE-compatible matplotlib figure style configuration."""

# IEEE single-column figure style
IEEE_STYLE = {
    "figure.figsize": (3.5, 2.625),
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.framealpha": 0.8,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.format": "pdf",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
}

# Standard markers and colors for up to 6 methods
MARKERS = ["o", "s", "^", "D", "v", "p"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--"]


def apply_ieee_style() -> None:
    """Apply IEEE figure style globally to matplotlib."""
    import matplotlib
    matplotlib.rcParams.update(IEEE_STYLE)
