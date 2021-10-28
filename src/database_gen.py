#!/bin/python3

import math
import numpy as np
import matplotlib.pyplot as plt
import gmsh
import sys

import procruste as pr


def mesh_contour(coord: np.ndarray, mesh_file, h: float) -> int:
    """Simple mesh création with Gmsh API


    """
    gmsh.initialize()

    gmsh.model.add("polygon")

    # Number of vertices in contour
    nb_v_in_c = len(coord)

    # Vertices
    for i in range(nb_v_in_c):
        x = coord[i, 0]
        y = coord[i, 1]
        gmsh.model.geo.addPoint(x, y, 0, h, i)

    # Edges
    for i in range(nb_v_in_c):
        gmsh.model.geo.addLine(i, (i+1) % nb_v_in_c, i)

    gmsh.model.geo.addCurveLoop([i for i in range(nb_v_in_c)], 1)

    gmsh.model.geo.addPlaneSurface([1], 1)

    gmsh.model.geo.synchronize()

    # Meshing
    gmsh.model.mesh.generate(2)

    # Number of vertices
    nb_v = len(gmsh.model.mesh.get_nodes()[0])

    # Number of inner_vertices
    nb_inner_v = nb_v - nb_v_in_c

    gmsh.write(mesh_file)

    # Open mesh in GUI
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()

    return nb_inner_v


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

    coord = create_random_contour(nvert=50)
    pr.procruste(coord)
    # Pour ne pas mailler les contours (ne pas les subdiviser), on prend h>>(longueur des contours)
    mesh_contour(coord, "polygon.mesh", 10)
    export_contours(3)

    return


if __name__ == "__main__":
    main()
