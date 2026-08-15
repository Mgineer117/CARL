import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_distinct_centered_data(mode, n_samples=3000):
    """
    Generates data (x, u) centered in the plot.
    """
    # np.random.seed(42)

    # Plot center coordinates
    c_x, c_u = 0.5, 0.5

    if mode == "real_world":
        # --- 1. Real-World: TINY, Centered Blob (Gaussian Mixture) ---
        cov1 = [[0.008, 0.004], [0.004, 0.008]]
        data1 = np.random.multivariate_normal([c_x - 0.02, c_u - 0.02], cov1, int(n_samples / 2))

        cov2 = [[0.002, -0.001], [-0.001, 0.002]]
        data2 = np.random.multivariate_normal([c_x + 0.02, c_u + 0.02], cov2, int(n_samples / 2))

        data = np.vstack((data1, data2))
        x = data[:, 0]
        u = data[:, 1]
        return x, u

    elif mode == "control_focused":
        # --- 2. Control-Focused: LARGE Tall/Thin Rectangle (Uniform) ---
        width = 0.3
        height = 0.9

        x_min = c_x - (width / 2)
        u_min = c_u - (height / 2)

        # REVISION: Used Uniform distribution for even fill
        x = np.random.uniform(x_min, x_min + width, n_samples)
        u = np.random.uniform(u_min, u_min + height, n_samples)
        return x, u

    elif mode == "state_focused":
        # --- 3. State-Focused: LARGE Wide/Short Rectangle (Uniform) ---
        width = 0.9
        height = 0.3

        x_min = c_x - (width / 2)
        u_min = c_u - (height / 2)

        # REVISION: Used Uniform distribution for even fill
        x = np.random.uniform(x_min, x_min + width, n_samples)
        u = np.random.uniform(u_min, u_min + height, n_samples)
        return x, u


def draw_final_plot(ax, x_data, u_data, title):
    """
    Draws the plot with expanded density fill and no dotted lines.
    """
    PLOT_MAX = 1.0

    # --- Draw Density Plot ---
    # bw_adjust=0.5: Sharper edges to fit the uniform shape
    # thresh=0.01:   Lowers the cutoff so the fill extends to the data limits
    sns.kdeplot(
        x=x_data,
        y=u_data,
        ax=ax,
        fill=True,
        cmap="Blues",
        alpha=0.9,
        levels=7,
        thresh=0.01,
        bw_adjust=0.5,
        zorder=2,
    )

    # --- Draw Contour Outline ---
    sns.kdeplot(
        x=x_data,
        y=u_data,
        ax=ax,
        fill=False,
        color="gray",
        alpha=0.3,
        linewidths=1.5,
        levels=[0.01],
        bw_adjust=0.5,
        zorder=3,
    )

    # --- Styling ---
    ax.set_xlim(0, PLOT_MAX)
    ax.set_ylim(0, PLOT_MAX)

    # Custom thick axes arrows
    ax.spines["left"].set_position(("data", 0))
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Arrowheads
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False, markersize=10)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, markersize=10)

    # Labels
    ax.set_xlabel(r"$|\mathcal{X}|$", fontsize=20, loc="right", weight="bold")
    ax.set_ylabel(r"$|\mathcal{U}|$", fontsize=20, loc="top", rotation=0, weight="bold")

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(title, fontsize=22, pad=20, weight="bold")


# --- Main Execution ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 0. Baseline (State-Focused)
x3, u3 = generate_distinct_centered_data("state_focused")
draw_final_plot(axes[0], x3, u3, "Baseline")

# 1. Control-Focused
x2, u2 = generate_distinct_centered_data("control_focused")
draw_final_plot(axes[1], x2, u2, "Control-focused")

# 2. Real-World
x1, u1 = generate_distinct_centered_data("real_world")
draw_final_plot(axes[2], x1, u1, "Real-world-focused")


plt.tight_layout()
plt.savefig("synthetic_data_final.svg", dpi=300)
plt.savefig("synthetic_data_final.pdf", dpi=300)
plt.show()
