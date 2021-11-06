import hashlib
import json
from argparse import Namespace
from pathlib import Path

import datashader as ds
import pandas as pd
from datashader import transfer_functions as tf
from datashader.utils import export_image
from numpy import save
from numpy.core.fromnumeric import searchsorted


def save_img(img, args: Namespace) -> None:
    folder = f"./plots/{args.attractor}"
    Path(folder).mkdir(parents=True, exist_ok=True)
    dict_str = str(vars(args))
    fn = hashlib.sha224(bytes(dict_str, "utf-8")).hexdigest()[:15]
    export_image(img, f"{folder}/{fn}", fmt=".png")
    save_dict = vars(args)
    save_dict["x_0"] = list(save_dict["x_0"])
    with open(f"{folder}/{fn}.json", "wt") as f:
        json.dump(save_dict, f, indent=4)


def plot_2d_attractor(df: pd.DataFrame, args: Namespace) -> None:
    cvs = ds.Canvas(plot_width=1800, plot_height=1800)
    agg = cvs.points(df, "x", "y")
    ds.transfer_functions.Image.border = 0
    img = tf.shade(agg, cmap=["white", "black"])

    save_img(img, args)
