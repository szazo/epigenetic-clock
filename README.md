# Assignment for Bioinformatics aspects of aging and rejuvenation

Epigenetic clock assignments for *Bioinformatics aspects of aging and rejuvenation* lecture.

Lecturer: Csaba Kerepesi

* Assignment1: Train epigenetic clock for microarray-based methylation dataset
* Assignment2: Train epigenetic clock for RRBS-based methylation dataset (with missing data)

Bioinformatics aspects of aging and rejuvenation

## Setup

## Creating and activating virtual env

1. Create a *virtual env* which uses Python 3.9 (**replace** `VENV_DIR` e.g. to `~/.envs/bio`):
    ```bash
    pyenv install 3.8
    pyenv shell 3.8
    python -m venv VENV_DIR
    ```
2. Activate the virtual env:
    ```bash
    source VENV_DIR/bin/activate
    ```
    
## Initialize the project for development

1. Enter into **project root**
2. Start developing with **editable** (or **develop**) mode:
    ```bash
    python -m pip install --editable ".[dev]"
    ```
3. Optional, install pre-commit hooks
    ```bash
    pre-commit install
    ```
4. Install python environment as Jupyter kernel
   ```bash
   python -m ipykernel install --user --name=bio
   ```
   
# Other goodies

## Tips for development

### Uninstall all packages (and purge pip cache)

```
pip uninstall epigenetic_clock_assignment
pip freeze | xargs pip uninstall -y
pip cache purge
```
