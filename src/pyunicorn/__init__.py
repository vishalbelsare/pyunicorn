# This file is part of pyunicorn.
# Copyright (C) 2008--2025 Jonathan F. Donges and pyunicorn authors
# URL: <https://www.pik-potsdam.de/members/donges/software-2/software>
# License: BSD (3-clause)
#
# Please acknowledge and cite the use of this software and its authors
# when results are used in publications or published elsewhere.
#
# You can use the following reference:
# J.F. Donges, J. Heitzig, B. Beronov, M. Wiedermann, J. Runge, Q.-Y. Feng,
# L. Tupikina, V. Stolbova, R.V. Donner, N. Marwan, H.A. Dijkstra,
# and J. Kurths, "Unified functional network and nonlinear time series analysis
# for complex systems science: The pyunicorn package"

"""
pyunicorn
=========

Subpackages
-----------
core
    Spatially embedded complex networks and multivariate data
climate
    Climate networks
funcnet
    Functional networks
timeseries
    Time series surrogates
"""

from .version import __version__
from .utils import mpi
from .core import *
