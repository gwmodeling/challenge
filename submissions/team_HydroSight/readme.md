# Team HydroSight

In this file the submission is described. 

## Author(s)

- Xinyang Fan (xinyangf1@student.unimelb.edu.au) Department of Infrastructure Engineering, The University of Melbourne
- Tim Peterson (tim.peterson@monash.edu) Department of Civil Engineering, Monash University 

## Modeled locations

We modelled the following locations (check modelled locations):

- [x] Netherlands
- [x] Germany
- [X] Sweden 1
- [x] Sweden 2
- [x] USA

## Model description
We applied a time-series conceptual groundwater model HydroSight (Peterson et al. 2014, Peterson et al. 2019) for this modelling challenge. The detailed description and scripts of the model are on https://github.com/peterson-tim-j/HydroSight. HydroSight can be run either independently as a script, or in the GUI which facilitates users to build up and calibrate the models. The calibrated and simulated models of this challenge can be open for viewing the results in the GUI.

## Model workflow to reproduce

1) Calculate the potential evapotranspiration (PET) with the provided climate variables using a range of methods available in the R package "Evapotranspiration" (Guo et al. 2016), see script PET_estimation.R
2) Snowmelt and snow storage are estimated with a Degree-Day factor method (Calli et al. 2022, JoH) which is also integrated in HydroSight. The DDF factor and the melting temperature (Tm) are the calibrated parameters.
3) Write the input files for HydroSight following the requirements on https://github.com/peterson-tim-j/HydroSight: in this challenge, climate files are the only inputs including the daily precipitaion, PET (and mean air temperature depending on if the snowmelt module is activated); coordinates files contain the dummy data; models are calibrated to the observed heads.
4) Build up and calibrate the models with HydroSight. Detailed settings of our models are given in HydroSightModelling.m. 
5) Model performance is evaluated with the Nash-Sutcliff efficiency (NSE), model structure is evaluated with Akaike Information Criteria (AIC).
6) Future simulations are done with the calibrated HydroSight models which take the provided climate inputs.

References:
1) Peterson, Tim J., and A. W. Western. "Nonlinear time‐series modeling of unconfined groundwater head." Water Resources Research 50.10 (2014): 8330-8355.
2) Peterson, Tim J., and Simon Fulton. "Joint estimation of gross recharge, groundwater usage, and hydraulic properties within HydroSight." Groundwater 57.6 (2019): 860-876.
3) Guo, Danlu, Seth Westra, and Holger R. Maier. "An R package for modelling actual, potential and reference evapotranspiration." Environmental Modelling & Software 78 (2016): 216-224.
4) Çallı, Süleyman Selim, et al. "Contribution of the satellite-data driven snow routine to a karst hydrological model." Journal of Hydrology 607 (2022): 127511.



## Supplementary model data used

No supplementary data are needed.

## Estimation of effort

Please provide an (rough) estimate of the time it took to develop the models (e.g., read in data, pick a model 
structure, etcetera) and calibrate the parameters (if any). If possible, please also state the computational resources that 
were required.

| Location    | Development time (hrs) | Calibration time (hrs) | Total time (hrs) | 
|-------------|------------------------|------------------------|------------------|
| Netherlands | ~ 5 mins               | 2                      | ~2.5             |
| Germany     | ~ 5 mins               | 2                      | ~2.5             |
| Sweden 1    | ~ 5 mins               | 2                      | ~2.5             |
| Sweden 2    | ~ 5 mins               | 2                      | ~2.5             |
| USA         | ~ 5 mins               | 2                      | ~2.5             |

## Additional information

Each model is calibrated with 6 cores CPU.
