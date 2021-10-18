import argparse
from typing import Dict

from attractors import Attractor, Attractors


def main(args: argparse.Namespace, optional_params: Dict[str, float]) -> None:
    attractor = Attractor.load(args.name, args.N, args.x_0, optional_params)
    print(attractor.step((0, 0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise strange attractors. Optional parameters can be provided, ex: --a 1.2"
    )
    parser.add_argument(
        "name",
        help="Name of the strange attractor",
        choices=[a.value for a in Attractors],
        type=str,
    )
    parser.add_argument(
        "x_0",
        type=float,
        nargs="+",
        help="Starting point for the simulation, space separated. Ex: 0 1.2",
    )
    parser.add_argument("--N", type=int, default=1000)

    args, rest = parser.parse_known_args()
    args.x_0 = tuple(args.x_0)
    optional_params = {
        rest[i].replace("--", ""): float(rest[i + 1]) for i in range(0, len(rest), 2)
    }

    main(args, optional_params)
