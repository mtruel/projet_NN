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

To understand what a neural network is, it is first necessary to understand what a neuron is in computer science. A neuron is an object that takes as input an array of data, that performs a specific operation on this data, and that outputs a single value. The most simple and used neurons are called linear, and perform a linear combination of the input data: a weight :math:`w_i` is associated to each piece of input data :math:`x_i` and the output is computed as :math:`y = \sum_{i=1}^n{w_i x_i} + b` where :math:`b` is a bias (it can be seen as a weight associated to the input value :math:`x_0 = 1`). To introduce non-linearity in the process, the output is then processed through an activation function :math:`f` (Heaviside, ReLU, Sigmoid...). A neuron is fully defined by the value of its weights :math:`w_i, i\in\{0,...,n\}`. A representation of a single linear neuron with the Heaviside activation function can be seen in figure :numref:`neuron`.

.. _neuron:
.. figure:: images/neuron.png
  :width: 4000
  :class: no-scaled-link
  :align: center
  :alt: Representation of a single linear neuron with the Heaviside activation function.

  Representation of a single linear neuron with the Heaviside activation function.

A neural network is a collection of neurons organized in layers, each layer containing a fixed number of neurons. The output of each layer is taken as the input of each neuron of the next layer. A neural network is fully defined by the value of the weights of its neurons, and, knowing the values of these weights, a single deterministic output can be computed knowing the input array.


.. _neural_network:
.. figure:: images/neural_network.svg
  :width: 4000
  :class: no-scaled-link
  :align: center
  :alt: Representation of a neural network.

  Representation of a neural network.

In order to obtain a desired output from a certain input, the weights must have "correct" values. To compute these values, the network has to be trained before it can be used. This training process requires a database of inputs and desired outputs. The inputs go through the neural networks, which produces an output that is compared to the desired output. Using some error measure between the predicted output and the desired output, the weights are then updated in order to minimize this error. The error measure is called the loss function, and the weights are updated using a gradient descent method aimed at minimizing this function. In general, the weights are not updated after each sample of the database because it's too cotsly in term of computational resources. Instead, the loss function is computed with a certain number of samples (50 for instance), and then the weights are updated by minimizing this loss. Such a group of samples is called a batch. Finally, a single run through the database is not enough to train the network, and the database must be run through many times. A full pass through the entire database is called an epoch. In general, many epochs are needed to train efficiently the neural network.


Meshing with Neural Networks ?
------------------------------

Since efficient classical meshing algorithms and have been widely studied and implemented, why should neural networks be used in the meshing process? The first reason is that, once the networks are trained, generating a mesh is expected to be very fast since it only consists in a series of simple operations on arrays of data. Therefore, one of the goal is to generate meshes of good quality faster than a Delaunay-based meshing software.
Another reason one might want to use neural network is automation. For very complex geometrical configurations, even with efficient automatic Delaunay-based meshers, userintervention can be necessary in some parts of the meshing process. Now, let's imagine that a neural network has been trained with such complex meshes, then one can expect the network to be able to generate meshes for complex configurations without any human intervention.
