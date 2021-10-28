#!/bin/python3

import math
import numpy as np
import matplotlib.pyplot as plt
import gmsh
import sys


def mesh_contour(coord: np.ndarray, mesh_file, h: float) -> None:
    """Simple mesh création with Gmsh API


    """
    gmsh.initialize()

    gmsh.model.add("polygon")

    nb_vertices = len(coord)

    # Vertices
    for i in range(nb_vertices):
        x = coord[i, 0]
        y = coord[i, 1]
        gmsh.model.geo.addPoint(x, y, 0, h, i)

    # Edges
    for i in range(nb_vertices):
        gmsh.model.geo.addLine(i, (i+1) % nb_vertices, i)

    gmsh.model.geo.addCurveLoop([i for i in range(nb_vertices)], 1)

    gmsh.model.geo.addPlaneSurface([1], 1)

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.write(mesh_file)

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()


def create_random_contour(nvert: int) -> np.ndarray:
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


def export_contours(n: int) -> None:

    for k in range(n):
        coord = create_random_contour(nvert=6)
        f = open("exports/export"+str(k)+".txt", "w+")
        for i in coord:
            f.write(str(i[0])+" ")
            f.write(str(i[1])+"\n")
    return 0


def main():

    coord = create_random_contour(nvert=6)
    #mesh_contour(coord, 1)
    export_contours(3)
    return


if __name__ == "__main__":
    main()
