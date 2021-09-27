# bslib - battery storage library

Repository with code to
 
- build a **database** with relevant data from PerMod database (HTW Berlin) and "Stromspeicher-Inspektion"
- **simulate** ac-, dc- and pv-generator coupled battery storages with regards to electrical power (ac and dc) and state-of-charge as timeseries.

For the simulation, it is possible to calculate outputs of a **specific manufacturer + model** or alternatively for one of **3 different generic battery types**. 

## Documentation

For a basic understanding of is *bslib* have a look into the Documentation [HTML] or [Jupyter-Notebook]. There you also find a **simulation examples**.

## Usage

Download or clone repository:

`git clone https://github.com/RE-Lab-Projects/bslib.git`

Create the environment:

`conda env create --name bslib --file requirements.txt`

Create some code with `import bslib` and use the included functions `load_database`, `get_parameters` and `simulate`.

## Battery models and Group IDs
The hplib_database.csv contains the following number of heat pump models, sorted by Group ID

| [Group ID]: Count | Description |
| :--- | :--- |
| AC-coupled | [..]: .. | [.: 23 |


## Database

All resulting database CSV file are under [![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/).

The following columns are available for every heat pump of this library

| Column | Description | Comment |
| :--- | :--- | :--- |
| .. | .. | .. |


## Input-Data and further development
...

**Further development | Possibilities to collaborate**

...

If you find errors or are interested in develop the hplib, please create an ISSUE and/or FORK this repository and create a PULL REQUEST.
