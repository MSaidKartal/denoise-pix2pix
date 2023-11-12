from ipywidgets import interact, IntSlider
from matplotlib.patches import PathPatch, Rectangle
import matplotlib.pyplot as plt
from matplotlib.path import Path


def _draw_line(ax, coords, clr='g'):
    line = Path(coords, [Path.MOVETO, Path.LINETO])
    pp = PathPatch(line, linewidth=3, edgecolor=clr, facecolor='none')
    ax.add_patch(pp)

def _set_axes_labels(ax, axes_x, axes_y):
    ax.set_xlabel(axes_x)
    ax.set_ylabel(axes_y)
    ax.set_aspect('equal', 'box')

_rec_prop = dict(linewidth=5, facecolor='none')

def show_volume(vol, z, fig_size=(14, 7)):
    #print(vol[0].shape)
    #print(vol[1].shape)

    fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=fig_size)
    v_z, v_y, v_x = vol[0].shape

    axarr[0].imshow(vol[0][z, :, :], cmap="gray")
    axarr[0].set_title('Low Resolution', fontweight='bold', fontsize=20)
    axarr[0].axis('off')

    axarr[1].imshow(vol[1][z, :, :], cmap="gray")
    axarr[1].set_title('High Resolution', fontweight='bold', fontsize=20)
    axarr[1].axis('off')

    fig.tight_layout()


def interactive_show(vol):
    volume= vol[0]
    vol_shape = volume.shape
    interact(
        lambda z: plt.show(show_volume(vol, z)),
        z=IntSlider(min=0, max=vol_shape[0], step=1, value=int(vol_shape[0] / 2)),

    )
