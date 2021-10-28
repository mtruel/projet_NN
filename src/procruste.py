#!/bin/python3

import numpy as np
import math
import scipy.spatial as sp
import matplotlib.pyplot as plt


def reg_unit_polygon_gen(nvert: int) -> np.ndarray:
    """

    """

    theta = 0.0
    coord = np.ndarray((nvert, 2))

    for i in range(nvert):
        theta = i * 2.0 * math.pi / float(nvert)
        coord[i, 0] = math.cos(theta)
        coord[i, 1] = math.sin(theta)

    return coord


def procruste(coord: np.ndarray) -> None:
    """

    """

    nvert = len(coord)
    reg_coord = reg_unit_polygon_gen(nvert)

    plt.plot(coord[:, 0], coord[:, 1])
    plt.plot(reg_coord[:, 0], reg_coord[:, 1])
    plt.show()

    # In the procruste algorithm, points are scaled with the absolute 2-norm,
    # but maybe it is better to scale with the 2-norm scaled by the number of points ???
    reg_coord, coord, disparity = sp.procrustes(reg_coord, coord)

    plt.plot(coord[:, 0], coord[:, 1])
    plt.plot(reg_coord[:, 0], reg_coord[:, 1])
    plt.show()
    return


def main():

    return


if __name__ == "__main__":
    main()
