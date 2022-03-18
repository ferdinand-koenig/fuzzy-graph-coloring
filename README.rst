fuzzy-graph-coloring
********************

fuzzy-graph-coloring is a Python package for calculating
the fuzzy chromatic number and coloring of a graph with fuzzy edges.
It will create a coloring with a minimal amount of incompatible edges
using a genetic algorithm (:code:`genetic_fuzzy_color`) or a greedy-k-coloring (:code:`greedy_k_color`)
combined with a binary search (:code:`alpha_fuzzy_color`).

If you don't know which one to use, we recommend :code:`alpha_fuzzy_color`.
If you are looking for a networkX coloring but with a given k, use :code:`greedy_k_color`.

See repository https://github.com/ferdinand-dhbw/fuzzy-graph-coloring

Quick-Start
===========
Install package: :code:`pip install fuzzy-graph-coloring`

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

   print(fgc.alpha_fuzzy_color(TG1, 3, return_alpha=True, fair=True))

Result: :code:`({1: 0, 4: 1, 2: 2, 3: 2}, 0.918918918918919, 0.4)`

(Tuple of coloring, score [(1-DTI)], and alpha [of alpha-cut])

Bibliography
============
The project uses a lot of the by Keshavarz created basics:
E. Keshavarz, "Vertex-coloring of fuzzy graphs: A new approach," Journal of Intelligent & Fuzzy Systems, vol. 30, pp. 883-893, 2016, issn: 1875-8967. https://doi.org/10.3233/IFS-151810

License
=======
This project is licensed under GNU General Public License v3.0 (GNU GPLv3). See :code:`LICENSE` in the code repository.


Setup development environment
=============================
1. Get poetry https://python-poetry.org/docs/
2. Make sure, Python 3.8 is being used
3. :code:`poetry install` in your system shell
4. :code:`poetry run pre-commit install`

Run pre-commit
--------------
:code:`poetry run pre-commit run --all-files`

Run pytest
----------
:code:`poetry run pytest .\tests`

Create documentation
--------------------
:code:`.\docs\make html`
