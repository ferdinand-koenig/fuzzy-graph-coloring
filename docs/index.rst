.. fuzzy-graph-coloring documentation master file, created by
   sphinx-quickstart on Tue Feb  8 14:25:54 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to fuzzy-graph-coloring's documentation!
************************************************

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

fuzzy-graph-coloring is a Python package for calculating
the fuzzy chromatic number and coloring of a graph with fuzzy edges.
It will create a coloring with a minimal amount of incompatible edges
using a genetic algorithm.

See repository https://github.com/ferdinand-dhbw/fuzzy-graph-coloring


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Quick-Start
===========
Install package: `pip install fuzzy-graph-coloring`

Try simple code:

.. code-block::

   import fuzzy-graph-coloring as fgc

   TG1 = nx.Graph()
   TG1.add_edge(1, 2, weight=0.7)
   TG1.add_edge(1, 3, weight=0.8)
   TG1.add_edge(1, 4, weight=0.5)
   TG1.add_edge(2, 3, weight=0.3)
   TG1.add_edge(2, 4, weight=0.4)
   TG1.add_edge(3, 4, weight=1.0)

   fgc.fuzzy_color(TG1, 3)

Result: `({1: 2, 2: 3, 3: 3, 4: 1}, 0.918918918918919)`

Public functions
================
.. automodule:: fuzzy_graph_coloring
   :members:

Bibliography
============
The project uses a lot of the by Keshavarz created basics:
E. Keshavarz, “Vertex-coloring of fuzzy graphs: A new approach,” Journal of Intelligent & Fuzzy Systems, vol. 30, pp. 883–893, 2016, issn: 1875-8967. https://doi.org/10.3233/IFS-151810

License
=======
Unsere Lizenz