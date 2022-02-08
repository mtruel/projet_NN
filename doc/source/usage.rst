Installation and usage
======================

In this section, the installation process is detailed and some information on how to use the code is given. Note that python3.9 is required for this software to work properly (mainly because of pytorch).

Get the sources
---------------

The sources are available on a GitHub repository that can be found `here <https://github.com/MathouLrt/projet_NN>`_. To get the sources, open a terminal and type

.. code-block:: bash

                git clone https://github.com/MathouLrt/projet_NN.git path/to/the/desired/location

Install and create the python environment
-----------------------------------------
This software needs some python modules in order to work properly. It was packaged using python virtual environments. To install the necessary packages and create the python's virtual environment, cd into the project's root directory and type

.. code-block:: bash

                make install
                source .env/bin/activate

Generating the database
-----------------------
You can now generate the database for the NN1 or NN2 by typing

.. code-block:: bash

                python3.9 ./src/database_gen.py -NNx

where x is 1 or 2 depending on what you need.

Training the networks
---------------------
To train the neural networks, just type

.. code-block:: bash

                python3.9 ./src/NNx.py

where x is 1 or 2 depending on what you need. At the end of the training process, the models (e.g. the networks parameters)  are saved into binary files by pytorch under the data directory. They can be retrieved in another python program using the pytorch API.

Ending your session
-------------------
When you are done with our software, type the following command to deactivate the python virtual environment

.. code-block:: bash

                deactivate
