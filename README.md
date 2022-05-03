# bslib - battery storage library

Repository with code to
 
- build a **database** with relevant data from PerMod database (HTW Berlin) and "Stromspeicher-Inspektion"
- **simulate** ac- and dc-coupled battery storages with regards to electrical power (ac and dc) and state-of-charge as timeseries.

For the simulation, it is possible to calculate outputs of a **specific manufacturer + model** or alternatively for one of **2 different generic battery types**. 

## Documentation

The documentation is still under development.

## Usage

Simply install via

- `pip install bslib`

or clone repository and create environment via:

- `git clone https://github.com/RE-Lab-Projects/bslib.git`
- `conda env create --name bslib --file requirements.txt`

Afterwards you're able to create some code with `import bslib` and use the included functions `load_database`, `get_parameters` and `simulate`.

## Battery models and Group IDs
The bslib_database.csv contains the following number of battery storages, sorted by Group ID

| [Group ID]: Count | Description |
| :--- | :--- |
| [S_ac]: 2 | AC-coupled |
| [S_dc]: 3 | DC-coupled |
| [INV]: 2 | PV Inverter |


## Database

All resulting database CSV file are under [![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/).

The following columns are available for every heat pump of this library

| Column | Description | Comment |
| :--- | :--- | :--- |
| .. | .. | .. |


## Input-Data and further development

If you find errors or are interested in develop the bslib, please create an ISSUE and/or FORK this repository and create a PULL REQUEST.
