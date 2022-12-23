# Team Example

In this file the submission is described. 

## Author(s)

- Didier VANDEN BERGHE ("VeloVolant")


## Modeled locations

We modelled the following locations (check modelled locations):

- [x] Netherlands
- [X] Germany
- [X] Sweden 1
- [X] Sweden 2
- [X] USA

## Model description

We used the model GARDENIA 8.8 as described in detail in Thiéry (2011) - BRGM (French Geological Survey). This is a Grey Box type of model. The model is 
implemented in the GARDENIA software package that was used here.

## Model workflow to reproduce

The model has a user interface that produces input files (txt format), explicit and easy to manipulate for any French speaking scientist. The model has the capability to simulate one or several reservoirs. Up to 3 reservoirs have been considered, for the NL case, however a single reservoir approach was the default and simplifiest strategy adopted as often as possible. The GARDENIA model relies on some simplified water budget and hydrodynamic parameters. However little consideration for the reliability of the parameter values was given, as the goal is more to perform a good "statistical" calibration and thus a good prediction rather than a good physical model calibration. 

## Supplementary model data used

No additional data were used. Only the gw levels, rr and et records were used.

## Estimation of effort

Please provide an (rough) estimate of the time it took to develop the models (e.g., read in data, pick a model 
structure, etcetera) and calibrate the parameters (if any). If possible, please also state the computational resources that 
were required.

| Location    | Development time (hrs) | Calibration time (s) | Total time (hrs) | 
|-------------|------------------------|----------------------|------------------|
| Netherlands | ~ 0,5                  | 40                   | 02:00:40         |
| Germany     | ~ 0,25                 | 40                   | 02:00:40         |
| Sweden 1    | ~ 0,5                  | 40                   | 02:00:40         |
| Sweden 2    | ~ 0,5                  | 40                   | 02:00:40         |
| USA         | ~ 0,25                 | 40                   | 02:00:40         |

## Additional information

Swedisch cases were not calibrated correctly (Nash around 3). NL case was calibrated correctly only if considering 3 reservoirs. 
