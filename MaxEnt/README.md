# README

## Overview

This project contains two Jupyter notebooks:

-   `run_subtask1_MaxEnt.ipynb`
-   `run_subtask2a_MaxEnt.ipynb`

These notebooks implement and run Maximum Entropy (MaxEnt) models for
two different subtasks.

------------------------------------------------------------------------

## Requirements

-   Python 3.8+
-   Jupyter Notebook or JupyterLab

Install required packages:

``` bash
pip install numpy pandas scikit-learn scipy matplotlib nltk jupyter
```

------------------------------------------------------------------------

## Project Structure

    .
    ├── run_subtask1_MaxEnt.ipynb
    ├── run_subtask2a_MaxEnt.ipynb
    ├── data/        # (if applicable) input datasets
    ├── models/      # (if applicable) saved models
    └── outputs/     # (if applicable) results

------------------------------------------------------------------------

## How to Run

### Option 1: Run with Jupyter

``` bash
jupyter notebook
```

Open both notebooks and click:

Kernel → Restart & Run All

------------------------------------------------------------------------

### Option 2: Run from Command Line

``` bash
jupyter nbconvert --to notebook --execute run_subtask1_MaxEnt.ipynb
jupyter nbconvert --to notebook --execute run_subtask2a_MaxEnt.ipynb
```

------------------------------------------------------------------------

## Execution Order

If Subtask 2a depends on Subtask 1 outputs:

1.  Run `run_subtask1_MaxEnt.ipynb`
2.  Then run `run_subtask2a_MaxEnt.ipynb`

------------------------------------------------------------------------

## Notes

-   Ensure all dataset paths inside notebooks are correct.
-   It is recommended to use a virtual environment:

``` bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
pip install numpy pandas scikit-learn scipy matplotlib nltk jupyter
```

------------------------------------------------------------------------

## Troubleshooting

-   Install missing packages using `pip install package_name`
-   Verify dataset file paths
-   Restart kernel if execution fails
