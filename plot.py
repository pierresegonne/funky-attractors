import hashlib
import json
from argparse import Namespace
from pathlib import Path

import datashader as ds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datashader import transfer_functions as tf
from datashader.utils import export_image
from numpy import save
from numpy.core.fromnumeric import searchsorted


def generate_filename(args: Namespace) -> str:
    folder = f"./plots/{args.attractor}"
    Path(folder).mkdir(parents=True, exist_ok=True)
    dict_str = str(vars(args))
    return f"{folder}/{hashlib.sha224(bytes(dict_str, 'utf-8')).hexdigest()[:15]}"


def save_params(filename: str, args: Namespace) -> None:
    save_dict = vars(args)
    save_dict["x_0"] = list(save_dict["x_0"].astype(float))
    with open(f"{filename}.json", "wt") as f:
        json.dump(save_dict, f, indent=4)


def save_img(filename: str, img) -> None:
    export_image(img, f"{filename}", background="white", fmt=".png")


def plot_2d_attractor(df: pd.DataFrame, args: Namespace) -> None:
    cvs = ds.Canvas(plot_width=1800, plot_height=1800)
    agg = cvs.points(df, "x", "y")
    ds.transfer_functions.Image.border = 0
    img = tf.shade(agg, cmap=["white", "black"])
    filename = generate_filename(args)
    save_img(filename, img)
    save_params(filename, args)


def plot_3d_attractor(x: np.array, args: Namespace) -> None:
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection="3d")
    plt.axis("off")
    plt.grid(b=None)

    plt.plot(x[:, 0], x[:, 1], x[:, 2], ".", color="black", alpha=0.2, markersize=0.2)

    filename = generate_filename(args)
    save_params(filename, args)

    plt.savefig(
        f"{filename}.png", dpi=800, pad_inches=0, bbox_inches="tight", facecolor="white"
    )
    plt.close()
