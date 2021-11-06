import argparse
from warnings import warn

import numpy as np
import pandas as pd

from attractors import Attractors, trajectory_steps
from plot import plot_2d_attractor, plot_3d_attractor


def main(attractor, args: argparse.Namespace) -> None:
    assert args.attractor in [a.value.name for a in Attractors]
    # Generate the trajectory
    x = trajectory_steps(
        attractor.step,
        args.x_0,
        int(args.N_steps),
        float(args.a),
        float(args.b),
        float(args.c),
        float(args.d),
        float(args.e),
        float(args.f),
    )
    if attractor.N == 2:
        df = pd.DataFrame(dict(x=x[:, 0], y=x[:, 1]))
        plot_2d_attractor(df, args)
    elif attractor.N == 3:
        plot_3d_attractor(x, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise strange attractors. Optional parameters can be provided, ex: --a 1.2"
    )
    parser.add_argument(
        "attractor",
        help="Name of the strange attractor",
        choices=[a.value.name for a in Attractors],
        type=str,
    )
    parser.add_argument("--N_steps", type=int, default=10000000)
    parser.add_argument("--a", default=None)
    parser.add_argument("--b", default=None)
    parser.add_argument("--c", default=None)
    parser.add_argument("--d", default=None)
    parser.add_argument("--e", default=None)
    parser.add_argument("--f", default=None)
    parser.add_argument(
        "--x_0",
        # type=Optional[str],
        help="Starting point for the simulation, space separated. Ex: '(0,1.2)'. Default (0,0)",
        default=None,
    )

    args = parser.parse_args()
    attractor = None
    for att in Attractors:
        if att.value.name == args.attractor:
            attractor = att.value
            break
    args.x_0 = (
        np.array(eval(args.x_0)) if args.x_0 is not None else np.zeros(attractor.N)
    )
    assert args.x_0.shape == (attractor.N,)
    if any([(getattr(args, idf) is None) for idf in attractor.default_params.keys()]):
        warn(
            f"Missing step param {list(attractor.default_params.keys())}, fallback to default: {attractor.default_params}"
        )
        for idf in attractor.default_params.keys():
            if getattr(args, idf) is None:
                setattr(args, idf, attractor.default_params[idf])
    # Set to float
    for idf in ["a", "b", "c", "d", "e", "f"]:
        if getattr(args, idf) is None:
            setattr(args, idf, 0.0)

    main(attractor, args)
