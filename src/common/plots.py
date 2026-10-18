import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREY, DARK, ACCENT = "#9a9a9a", "#3a3a3a", "#c0392b"
GROUP_COLOURS = {"high": DARK, "medium": GREY, "low": ACCENT}


def new_axes(ncols=1, width=5.0, height=3.4):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt.subplots(1, ncols, figsize=(width * ncols, height))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
