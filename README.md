# nonmarkov-timeseries-estimation

This repository contains the code for implementing the framework presented in "A Modularized Efficient Framework for Non-Markov Time Series Estimation". We include jupyter notebooks for illustrative examples and recreation of figures.

All code of consequence can be found in the **python** directory. The file `framework.py` contains the underlying structure for setting up an estimation procedure that will be solved using ADMM. The file `updates.py` contains various implementations of update functions (separated by measurement model update, system model update, consensus update, and lagrange multiplier updates) for the problem formulations implemented so far. The file `helpers.py` contains additional functions that are useful and potentially reusable for the various updates. The remaining files serve to piece together updates with the framework to simplify the code that is written in the notebooks.

The **notebooks** directory contains iPython notebooks that are used to show how the solutions can be used. These notebooks were used to generate the figures and table in section IV of the paper.