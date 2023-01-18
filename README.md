[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6514527.svg)](https://doi.org/10.5281/zenodo.6514527)

<a href="https://www.fz-juelich.de/en/iek/iek-3"><img src="https://raw.githubusercontent.com/OfficialCodexplosive/README_Assets/862a93188b61ab4dd0eebde3ab5daad636e129d5/FJZ_IEK-3_logo.svg" alt="FZJ Logo" width="300px"></a>

# bslib - battery storage library

Repository with code to
 
- build a **database** with relevant data from PerMod database (HTW Berlin) and "Stromspeicher-Inspektion"
- **simulate** ac- and dc-coupled battery storages with regards to electrical power (ac and dc) and state-of-charge as timeseries.

For the simulation, it is possible to calculate outputs of a **specific manufacturer + model** or alternatively for one of **2 different generic battery storage types**. 

**For reference purposes:**
- DOI: https://doi.org/10.5281/zenodo.6514527
- Citation: Kai Rösken, Tjarko Tjaden, & Hauke Hoops. (2022). FZJ-IEK3-VSA/bslib: v0.7. Zenodo. https://doi.org/10.5281/zenodo.6514527

## Documentation

The documentation is still under development.

## Usage

Simply install via

- `pip install bslib`

or clone repository and create environment via:

- `git clone https://github.com/FZJ-IEK3-VSA/bslib.git`
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

The following columns are available for every battery storage of this library

| Column | Description | Comment |
| :--- | :--- | :--- |
| .. | .. | .. |


## Input-Data and further development

If you find errors or are interested in develop the bslib, please create an ISSUE and/or FORK this repository and create a PULL REQUEST.

## License
MIT License

Copyright (c) 2022

You should have received a copy of the MIT License along with this program.
If not, see https://opensource.org/licenses/MIT

## About Us
<p align="center"><a href="https://www.fz-juelich.de/en/iek/iek-3"><img src="https://github.com/OfficialCodexplosive/README_Assets/blob/master/iek3-wide.png?raw=true" alt="Institut TSA"></a></p>
We are the <a href="https://www.fz-juelich.de/en/iek/iek-3">Institute of Energy and Climate Research - Techno-economic Systems Analysis (IEK-3)</a> belonging to the <a href="https://www.fz-juelich.de/en">Forschungszentrum Jülich</a>. Our interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and costs of energy systems. The results are used for performing comparative assessment studies between the various systems. Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis studies for integrating new technologies into future energy market frameworks.
