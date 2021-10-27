#!/bin/python3

import math
import numpy as np
import matplotlib.pyplot as plt
import gmsh




def create_random_contour(nvert: int)->np.ndarray:
    """Create a random polygonal contour with nvert vertices
    
    :param nvert: The number of desired vertices
    :type nvert: int
    
    :return coord: Vector of coordinates for the nvert vertices
    :rtype type: np.ndarray
    """
    r = 1.
    rmin = 0.3
    theta = 0.
    x = y = 0.
    coord = np.ndarray((nvert, 2))

    for i in range(0, nvert):
        theta = (i + np.random.rand()) / nvert * 2 * math.pi
        r = np.random.rand() * (1 - rmin) + rmin
        x = r * math.cos(theta)
        y = r * math.sin(theta)

        coord[i, 0] = x
        coord[i, 1] = y
    return coord


def main():

    create_random_contour(nvert=100)

    return


if __name__ == "__main__":
    main()
