Motivations and theory
======================

An introduction on meshes and their use
---------------------------------------

Meshes are essential building blocks in most Computational Fuild Dynamics or Computational Solid Mechanics simulations. They allow to build a discrete version of the continuous problem that can be then solved using appropriate methods called numerical schemes.

Meshes are divided into two main categories : structured and unstructured meshes. In a structured mesh, all vertices have the same number of neighbours and the connectivity can be defined implicitly with simple relationships. In an unstructured mesh, the vertices can have a different amount of neighbours, and the connectivity between the elements has to be stored in an adequate data structure.

Meshes can also be simplicial or not. A simplicial mesh is a mesh containing only simplicial elements (triangles in 2D, tetrahedra in 3D). A mesh containing elements of different nature (triangles and quadrilaterals in 2D for instance) is called a mixed mesh.

A mesh is said to be conforming if every internal edge is shared by exactly two elements (boundary edges can only be shared by one element). A mesh that does not respect this criterion is said to be non-conforming.

In order to be usable for numerical schemes, a mesh has to meet certain requirements in term of topology and quality. These requirements directly depend on the actual physical problem that needs to be solved, on the numerical scheme that is used, and on the desired precision and/or ressource consumption of the simulation. In this work, the focus is put on simplicial and conforming unstructured meshes in 2D. These types of mesh are the most frequently used in CFD and CSM applications.

Many mesh generation algorithms have been developped and implemented efficiently in mesh generation softwares. These algorithms are generally divided into three main families : Delaunay-based methods, advancing front methods and quadtree/octree methods. The first two methods are able to produce very high quality meshes, can handle complex geometries and user-defined constraints (size-map, quality constraints...), but must be implemented efficiently and carefully. When implemented efficiently, Delaunay-based and advancing front methods have a time complexity of :math:`\mathcal{O}(n log(n))` with respect to the number of vertices in the final mesh. Quadtree/octree-based methods are generally easier to program and faster to run (time complexity in :math:`\mathcal{O}(n)`), but they are less versatile (they cannot handle complex geometries) and generally need further optimization steps in order to produce a quality mesh.

Several multi-threaded mesh generation algorithms have been developped and allow for a faster mesh generation. However, the introduction of parallelism into mesh generation algorithms is very hard and extra care must be taken when developping and implenting the algorithms in order to decrease the runtime while maintaining valid and good quality meshes.


An introduction on Neural Networks
----------------------------------




Meshing with Neural Networks ?
------------------------------

