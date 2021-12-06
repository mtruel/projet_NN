#!/bin/python3

import math
# from re import L
import numpy as np

import gmsh
import sys

from pathlib import Path
import shutil
import os
from tqdm import tqdm
import procrustes as pr


def mesh_contour(coord: np.ndarray, mesh_file) -> int:
    """Simple .mesh generation with Gmsh API from a given contour

    :param np.ndarray coord: The coordinates of the contour
    :param string mesh_file: The name of the output .mesh file

    :return: Number of inner vertices
    :rtype: int
    """
    # gmsh.initialize()

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
    # gmsh.model.mesh.setAlgorithm(2, 1, 3) #Add non points
    gmsh.model.mesh.generate(2)

    # Number of vertices
    nb_v = len(gmsh.model.mesh.getNodes()[0])

    # Number of inner_vertices
    nb_inner_v = nb_v - nb_v_in_c

    gmsh.write(str(mesh_file))

    # # Open mesh in GUI
    # if '-nopopup' not in sys.argv:
    #     gmsh.fltk.run()

    gmsh.model.remove()
    # gmsh.finalize()

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


def gen_database(Nc: int,  # Number of contour edges
                 # Dictionnary of requested polygons
                 # Request fomating dict({(ls,nb_of_polygons),(ls,nb_of_polygons)....})
                 requested_polygons: dict,
                 # Delete any previous files to start clean
                 # Data main folder
                 data_path: Path = Path("data"),
                 # Subfolders
                 meshes_folder: Path = Path("meshes"),
                 polygons_folder: Path = Path("polygons"),
                 # Label file
                 label_filename: Path = Path("labels"),
                 clean_data_dirs: bool = True) -> None:
    """Generates transformed contours and saves multiples output files 
    Takes a dictionnary with numbers of requested polygons
    Request fomating : dict({(ls,nb_of_polygons),(ls,nb_of_polygons)....})


    Creates folders to store meshes and polygons
    Creates label file with two cols : filename, number of inner vertices 
    Creates a file .dat for each polygon containing in one cols : (ls,x1,y1,x2,y2.....)^t
    Creates a file .mesh for each polygon generated by Gmsh

    Displays a simple progress bar with tqdm

    :param int Nc: Number of contour edges
    :param dict requested_polygons: Requested polygons 
    :param Path data_path: main folder to store dataset
    :param Path meshes_folder: folder to store .mesh files 
    :param Path polygons_folder:folder to store .dat files
    :param Path label_filename:Label file with 
    :param bool clean_data_dirs: Delete any previous file and directories

    :return: Creates the output files in the desired folder
    :rtype:
    """
    print(f"Generating database for {Nc} vertices.")

    # Add polygons folder
    data_path = data_path / Path(str(Nc))

    # Delete any previous files to start clean
    if clean_data_dirs:
        shutil.rmtree(data_path, ignore_errors=True)

    # create tree
    try:
        os.makedirs(data_path)
    except FileExistsError:
        pass
    # data_path.mkdir(exist_ok=True)
    (data_path / polygons_folder).mkdir(exist_ok=True)
    (data_path / meshes_folder).mkdir(exist_ok=True)

    gmsh.initialize()

    # Create label file
    with open(data_path / label_filename, "w+") as label_file:
        # Header
        label_file.write("filename, N1\n")
        # Generate polygons, tqdm create a progress bar
        idx = 0
        for ls in sorted(requested_polygons.keys()):
            for _ in tqdm(range(requested_polygons[ls])):
                polygon_filename = Path(f"poly{ls}_{idx}.dat")

                # Creation of polygon
                coord = create_random_contour(Nc)
                # Normalisation
                pr.procrustes(coord)
                # Mesh polygon and get nb of inner vertices
                nb_inner_vert = mesh_contour(
                    coord, data_path / meshes_folder / polygon_filename.with_suffix(".mesh"))

                # Write label files
                label_file.write(f"{polygon_filename}, {nb_inner_vert}\n")

                # Write polygon file
                with open(data_path / polygons_folder / polygon_filename, "w+") as polygon_file:
                    polygon_file.write(str(ls)+"\n")
                    for i in coord:
                        polygon_file.write(str(i[0])+"\n")
                        polygon_file.write(str(i[1])+"\n")
                idx += 1
    gmsh.finalize()
    return


def main():
    # Test one mesh
    # gmsh.initialize()
    # coord = create_random_contour(10)
    # pr.procrustes(coord)
    # mesh_contour(coord, "out.msh")
    # gmsh.finalize()

    # Gen database
    # request fomating dict({(ls,nb_of_polygons),(ls,nb_of_polygons)....})
    request = dict({(1.0, 6000)})
    gen_database(4, request)
    request = dict({(1.0, 12000)})
    gen_database(6, request)
    request = dict({(1.0, 24000)})
    gen_database(8, request)
    request = dict({(1.0, 48000)})
    gen_database(10, request)
    request = dict({(1.0, 95000)})
    gen_database(12, request)
    request = dict({(1.0, 190000)})
    gen_database(14, request)
    request = dict({(1.0, 380000)})
    gen_database(16, request)
    return


if __name__ == "__main__":
    main()
