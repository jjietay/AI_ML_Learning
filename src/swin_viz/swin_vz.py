"""
Swin Transformer — visual intuition builder.
=============================================

Draws four panels that mirror the actual operations at the start of a Swin
Transformer, so the picture you see *is* the tensor op:

  1. Original image
  2. Patch partition      -> reshape into PATCH_SIZE x PATCH_SIZE patches (+ zoom)
  3. Window partition     -> group patches into WINDOW_SIZE x WINDOW_SIZE windows
                             (W-MSA: self-attention happens *inside* each window)
  4. Shifted windows      -> np.roll the feature map by WINDOW_SIZE//2 patches,
                             then re-window (SW-MSA: lets neighbouring windows talk)

Everything is parameterised at the top. Change the three constants and re-run
to see how the grids change.

Run:  python swin_viz.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --------------------------------------------------------------------------- #
# Swin's real defaults. Image side must be divisible by PATCH_SIZE*WINDOW_SIZE
# for windows to tile evenly (here 4*7 = 28, and 224 = 8 * 28).
# --------------------------------------------------------------------------- #
IMG_SIZE    = 224          # resize the image to this square size (pixels)
PATCH_SIZE  = 4            # Swin partitions the input into 4x4 patches
WINDOW_SIZE = 7            # window = 7x7 patches  (M in the paper)
SHIFT       = WINDOW_SIZE // 2   # cyclic shift = 3 patches, used in SW-MSA

PATCH_PX  = PATCH_SIZE                       # 1 patch  = 4  px
WINDOW_PX = WINDOW_SIZE * PATCH_SIZE         # 1 window = 28 px
SHIFT_PX  = SHIFT * PATCH_SIZE               # shift    = 12 px


def load_image():
    """A built-in sample image, resized to a clean square (no network needed)."""
    from skimage import data
    from skimage.transform import resize
    img = data.astronaut()                          # 512x512x3 uint8
    img = resize(img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    return img


def draw_grid(ax, step, color, lw, ls="-", alpha=1.0):
    """Draw vertical+horizontal lines every `step` pixels."""
    for x in range(0, IMG_SIZE + 1, step):
        ax.axvline(x - 0.5, color=color, lw=lw, ls=ls, alpha=alpha)
    for y in range(0, IMG_SIZE + 1, step):
        ax.axhline(y - 0.5, color=color, lw=lw, ls=ls, alpha=alpha)


def tint_windows(ax, shift_px=0):
    """Overlay a translucent colour on every window so the partition is obvious."""
    cmap = plt.colormaps["tab20"]
    n = IMG_SIZE // WINDOW_PX            # windows per row (=8 with defaults)
    idx = 0
    # Start offset back by the shift so the *shifted* origin is shown correctly.
    start = -shift_px
    ys = list(range(start, IMG_SIZE, WINDOW_PX))
    xs = list(range(start, IMG_SIZE, WINDOW_PX))
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            color = cmap((idx % 20) / 20)
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), WINDOW_PX, WINDOW_PX,
                                    facecolor=color, edgecolor="white",
                                    lw=1.5, alpha=0.30))
            idx += 1


def main():
    img = load_image()
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Swin Transformer: patches -> windows -> shifted windows",
                 fontsize=16, fontweight="bold")

    # 1) ORIGINAL ----------------------------------------------------------- #
    ax = axes[0, 0]
    ax.imshow(img)
    ax.set_title(f"1. Input image  ({IMG_SIZE}x{IMG_SIZE})")

    # 2) PATCH PARTITION ---------------------------------------------------- #
    ax = axes[0, 1]
    ax.imshow(img)
    draw_grid(ax, PATCH_PX, color="cyan", lw=0.4, alpha=0.7)
    n_patches = IMG_SIZE // PATCH_PX
    ax.set_title(f"2. Patch partition\n{PATCH_SIZE}x{PATCH_SIZE} patches "
                 f"-> {n_patches}x{n_patches} grid")
    # zoom inset on the top-left corner so individual patches are visible
    axins = ax.inset_axes([0.62, 0.62, 0.36, 0.36])
    axins.imshow(img)
    draw_grid(axins, PATCH_PX, color="cyan", lw=0.8, alpha=0.9)
    axins.set_xlim(0, WINDOW_PX)            # show a few patches
    axins.set_ylim(WINDOW_PX, 0)
    axins.set_xticks([]); axins.set_yticks([])
    for s in axins.spines.values():
        s.set_edgecolor("yellow"); s.set_linewidth(2)
    ax.indicate_inset_zoom(axins, edgecolor="yellow")

    # 3) WINDOW PARTITION (W-MSA) ------------------------------------------- #
    ax = axes[1, 0]
    ax.imshow(img)
    tint_windows(ax, shift_px=0)
    draw_grid(ax, WINDOW_PX, color="white", lw=2.0)
    n_win = IMG_SIZE // WINDOW_PX
    ax.set_title(f"3. Window partition (W-MSA)\n{WINDOW_SIZE}x{WINDOW_SIZE} "
                 f"patches/window -> {n_win}x{n_win} windows\n"
                 f"attention is computed *within* each colour")

    # 4) SHIFTED WINDOWS (SW-MSA) ------------------------------------------- #
    ax = axes[1, 1]
    # the cyclic shift IS np.roll on the feature map:
    shifted = np.roll(img, shift=(-SHIFT_PX, -SHIFT_PX), axis=(0, 1))
    ax.imshow(shifted)
    tint_windows(ax, shift_px=0)            # regular windows on the rolled image
    draw_grid(ax, WINDOW_PX, color="white", lw=2.0)
    # mark the wrap-around seam (content from the top/left wrapped to bottom/right)
    seam = IMG_SIZE - SHIFT_PX
    ax.axvline(seam - 0.5, color="red", lw=2, ls="--")
    ax.axhline(seam - 0.5, color="red", lw=2, ls="--")
    ax.set_title(f"4. Shifted windows (SW-MSA)\ncyclic-shift by {SHIFT} patches "
                 f"({SHIFT_PX}px), then re-window\nred dashes = wrap-around seam")

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "swin_visualization.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()