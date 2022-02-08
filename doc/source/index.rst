.. Meshing with Neural Networks documentation master file, created by
   sphinx-quickstart on Thu Oct 28 13:41:33 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Meshing with Neural Networks
============================

Abstract 
--------------------------
The goal of this project is to implement an algorithm capable of creating 2D simplicial unstructured meshes of simple polygonal geometries using Neural Networks. This algorithm uses three neural networks called NN1, NN2 and NN3 in order to mesh the geometry, each one fulfilling a specific task. NN1 predicts the number of vertices to be inserted inside the polygon, NN2 predicts the location of those vertices, and NN3 creates the connectivity between the vertices. This algorithm is data driven: the neural networks have to be trained before they can be used. The training is done using a database of polygons that were meshed with an open-source Delaunay mesher.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   motivations
   algo_overview
   database_gen
   NN1
   NN2
   usage
   bibliography
   conclusion
   


Credits
-------
All credits go to Geoffrey Lebaud, Gabriel Suau, Lucas Trautmann and Mathias Truel. The project was proposed and supervised by Nicolas Barral.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
