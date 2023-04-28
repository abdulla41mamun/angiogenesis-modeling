from cell import Cell, CellMatrix, CellType
from vegf import VEGFMatrix

from typing import List, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from datetime import datetime
import gc


# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    fig = plt.figure()
    fig.set_size_inches(data.shape[0], data.shape[1])

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)


    cbar = ax.figure.colorbar(im, ax=ax, cax=cax, **cbar_kw)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=14)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    # ax.set_xticklabels(fontsize=14)
    # ax.set_yticklabels(fontsize=14)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False, labelsize=14)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor", fontsize=14)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar, fig, ax


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_cell_matrix(cell_matrix: CellMatrix, iteration: int, file_name: str = None):
    """Plot cell matrix at specific iteration """

    path = "plots/cell_matrix/"
    if file_name is None:
        file_name = str(datetime.now().strftime("%Y-%m-%d_%H%M"))
    path = path + file_name + '/'
    Path(path).mkdir(parents=True, exist_ok=True)
    name = file_name + "_cell_matrix_it" + str(iteration) + ".png"

    data = np.zeros((cell_matrix.height, cell_matrix.width))
    data_str = np.empty((cell_matrix.height, cell_matrix.width), dtype=object)
    for i in range(cell_matrix.height):
        for j in range(cell_matrix.width):
            obj = cell_matrix.matrix[i, j]
            if type(obj) == Cell and obj.cell_type == CellType.PHALANX:
                data[i, j] = CellType.PHALANX.value[0]
                data_str[i, j] = CellType.PHALANX.name[0]
            elif type(obj) == Cell and obj.cell_type == CellType.STALK:
                data[i, j] = CellType.STALK.value[0]
                data_str[i, j] = CellType.STALK.name[0]
            elif type(obj) == Cell and obj.cell_type == CellType.TIP:
                data[i, j] = CellType.TIP.value[0]
                data_str[i, j] = CellType.TIP.name[0]
            else:
                data_str[i, j] = '-'

    qrates = []
    for i in range(4):
        if i == 0:
            qrates.insert(0, '-')
        else:
            qrates.insert(0, CellType((i,)).name[0])

    norm = matplotlib.colors.BoundaryNorm(np.linspace(-0.5, 3.5, 5), 4)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: qrates[::-1][norm(x)])

    im, _ , fig, ax = heatmap(data,
                    cmap=plt.get_cmap("YlGnBu", 4), norm=norm,
                    cbar_kw=dict(ticks=np.arange(4), format=fmt),
                    cbarlabel="Cell Type")

    annotate_heatmap(im, valfmt=fmt, size=14, fontweight="bold", threshold=-1,
                     textcolors=("black", "white"))

    plt.savefig(path+name, bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close('all')
    plt.close()


    # im.cla()
    # im.clf()
    # im.close('all')

    # ax.cla()
    # ax.clf()
    # ax.close('all')

    fig.clear()
    plt.close(fig)

    gc.collect()


def plot_vegf_matrix(vegf_matrix: VEGFMatrix, iteration: int, file_name: str = None):
    """Plot VEGF matrix at specific iteration """
    path = "plots/vegf_matrix/"
    if file_name is None:
        file_name = str(datetime.now().strftime("%Y-%m-%d_%H%M"))
    path = path + file_name + '/'
    Path(path).mkdir(parents=True, exist_ok=True)
    name = file_name + "_vegf_matrix_it" + str(iteration) + ".png"

    data = vegf_matrix.matrix

    im, _ , fig, ax = heatmap(data, vmin=0,
                    cmap="magma_r", cbarlabel="VEGF")
    annotate_heatmap(im, valfmt="{x:.2f}", size=12, threshold=20,
                     textcolors=("red", "white"))

    plt.savefig(path+name, bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close('all')
    plt.close()


    # im.cla()
    # im.clf()
    # im.close('all')

    # ax.cla()
    # ax.clf()
    # ax.close('all')

    fig.clear()
    plt.close(fig)

    gc.collect()


def count_cells(cell_matrix: CellMatrix):
    "Counts number of each type of cell ofr cell matrix "

    phalanx_count = 0
    stalk_count = 0
    tip_count = 0

    for i in range(cell_matrix.height):
        for j in range(cell_matrix.width):
            obj = cell_matrix.matrix[i, j]
            if type(obj) == Cell and obj.cell_type == CellType.PHALANX:
                phalanx_count += 1
            elif type(obj) == Cell and obj.cell_type == CellType.STALK:
                stalk_count += 1
            elif type(obj) == Cell and obj.cell_type == CellType.TIP:
                tip_count += 1

    return phalanx_count, stalk_count, tip_count


def plot_cell_counts(data: List, file_name: str = None):
    path = "plots/"
    if file_name is None:
        file_name = str(datetime.now().strftime("%Y-%m-%d_%H%M"))
    Path(path).mkdir(parents=True, exist_ok=True)

    name = file_name + "_cell_counts.png"

    phalanx = [x[0] for x in data]
    stalk = [x[1] for x in data]
    tip = [x[2] for x in data]
    time = [x for x in range(len(data))]

    fig = plt.figure()
    plt.plot(time, phalanx, label="Phalanx")
    plt.plot(time, stalk, label="Stalk")
    plt.plot(time, tip, label="Tip")
    plt.xlabel('Time [h]')
    plt.ylabel('Number of cells [-]')
    plt.legend()

    plt.savefig(path+name, bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close('all')
    plt.close()

    fig.clear()
    plt.close(fig)

    gc.collect()








