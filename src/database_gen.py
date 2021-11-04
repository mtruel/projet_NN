#!/bin/python3

import math
import numpy as np
import matplotlib.pyplot as plt
import gmsh
import sys

import procrustes as pr


def mesh_contour(coord: np.ndarray, mesh_file) -> int:
    """Simple .mesh generation with Gmsh API from a given contour

    :param np.ndarray coord: The coordinates of the contour
    :param string mesh_file: The name of the output .mesh file

    :return: Number of inner vertices
    :rtype: int
    """
    gmsh.initialize()

    # Print only gmsh warnings and errors
    gmsh.option.setNumber("General.Verbosity", 2)

    gmsh.model.add("polygon")

    # Number of vertices in contour
    nb_v_in_c = len(coord)

    # Constraint (h >> contour_lenght to avoid meshing (subdividing) of contours)
    h = 10

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
    nb_v = len(gmsh.model.mesh.getNodes()[0])

    # Number of inner_vertices
    nb_inner_v = nb_v - nb_v_in_c

    gmsh.write(mesh_file)

    # # Open mesh in GUI
    # if '-nopopup' not in sys.argv:
    #     gmsh.fltk.run()

    gmsh.finalize()

    return nb_inner_v


def create_random_contour(nvert: int) -> np.ndarray:
    """Create a random polygonal contour with nvert vertices

    :param int nvert: The number of desired vertices

    :return: Vector of coordinates for the nvert vertices
    :rtype: np.ndarray
    """
    r = 1.
    rmin = 0.7
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
    """Exports the coordinates of randomly generated contours in multiple txt files

    :param int n: The number of desired text files

    :return: Creates the files in the desired folder
    :rtype:
    """

    for k in range(n):
        coord = create_random_contour(nvert=6)
        f = open("exports/export"+str(k)+".txt", "w+")
        for i in coord:
            f.write(str(i[0])+" ")
            f.write(str(i[1])+"\n")
    return 0


def main():

    coord = create_random_contour(nvert=50)
    pr.procrustes(coord)
    mesh_contour(coord, "polygon.mesh")
    # export_contours(3)

    return


if __name__ == "__main__":
    main()
