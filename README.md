# 2023W-SOS-A3 - LabelSOM visualization

The [labelsom notebook](./labelsom.ipynb) can be found in an HTML and PDF version.  
__Attention!__: Colors may not be shown correctly in the PDF export.

- [labelsom notebook as HTML](./SOS2023_ExSOM_group_coding_topic_d_00828589_01503441_01634039.html)
- [labelsom notebook as PDF](./SOS2023_ExSOM_group_coding_topic_d_00828589_01503441_01634039.pdf)

# Prerequisites

This repository contains `PySOMVis` as a git submodule.
Please ensure that the submodule is properly initiated and up-to-date by executing
```
git submodule update --init --recursive
```

The `PySOMVis` submodule is a slightly modified version of `PySOMVis` which can be 
installed as a Python package.

## Setup development environment

1. Setup Python venv and install necessary dependencies:

```
make dev
```

2. Install `labelsom` and `PySOMVis` libraries from within this repository:

```
make install
```

3. Ensure newly added code is formatted

```
make lint
```

# The `labelsom` library

This repository provides the source code for the `labelsom` Python library.
`labelsom` implements a SOM visualization similar the [Java SOMToolbox LabelSOM](https://www.ifs.tuwien.ac.at/dm/somtoolbox/somtoolbox-reference.html#LabelSOM).

1. Start `jupyter` to use the IPython notebook

```
jupyter-lab ./labelsom.ipynb
```

The jupyter server will be running at `http://localhost:8888/`
