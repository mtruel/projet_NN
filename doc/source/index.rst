.. Meshing with Neural Networks documentation master file, created by
   sphinx-quickstart on Thu Oct 28 13:41:33 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Meshing with Neural Networks
============================

Abstract 
--------------------------
In this project the goal is to implement an algorithm capable of meshing simple polygons. The method is data driven, meaning a database of thousand of polygons is used to train the algorithm. Once the model is trained, it is used to predict new data. Models are three neural networks called NN1 NN2 and NN3 with different goals. NN1 predicts the number of nodes to insert in the polygon. NN2 places the nodes. Finally, NN3 creates the connectivity between nodes. 


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   motivations
   algo_overview
   database_gen
   NN1
   NN2
   bibliography
   


Credits
-------
All credits go to Geoffrey Lebaud, Gabriel Suau, Lucas Trautmann and Mathias Truel. The project was proposed and supervised by Nicolas Barral.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
