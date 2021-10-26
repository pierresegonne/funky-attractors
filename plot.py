import datashader as ds
import pandas as pd
from datashader import transfer_functions as tf
from datashader.utils import export_image


def plot_2d_attractor(df: pd.DataFrame):
    cvs = ds.Canvas(plot_width=300, plot_height=300)
    agg = cvs.points(df, "x", "y")
    ds.transfer_functions.Image.border = 0
    img = tf.shade(agg, cmap=["white", "black"])

    export_image(img, "test")
