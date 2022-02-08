Conclusion
==================

We suceeded in implenting the first neural network which guesses the number of 
inner nodes in a given polygon. The second network which has to guess the location 
of these nodes has still to be finished.  
The third network has yet to be implemented.

Once these networks are working, the mesh generation seems to be promising 
on simple geometries. This algorithm could be useful for some specific usage 
such as meshing a large number of simple geometries, or a more complex 
geometry divided into smaller polygons. 

However, neural networks seem to be very constraining for the basic meshing 
practice: a new neural network has to be trained for each type of polygon, 
which requires a lot of time and memory.