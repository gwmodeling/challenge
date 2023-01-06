# Team runwaygrey

In this file the submission is described. 

## Author(s)

- Ayush Prasad (University of Helsinki)


## Modeled locations

We modelled the following locations (check modelled locations):

- [x] Netherlands
- [x] Germany
- [X] Sweden 1
- [x] Sweden 2
- [ ] USA

## Model description

We trained a single model for all the 4 locations using Mixed Effects Random Forest. This model is described in the paper http://www.tandfonline.com/doi/abs/10.1080/00949655.2012.741599 and implemented here https://github.com/manifoldai/merf

## Model workflow to reproduce

Pleae follow the Notebook gw_merf.ipynb

## Supplementary model data used

No additional information was obtained and/or used.

## Estimation of effort

Please provide an (rough) estimate of the time it took to develop the models (e.g., read in data, pick a model 
structure, etcetera) and calibrate the parameters (if any). If possible, please also state the computational resources that 
were required.

A single model was trained for all the sites together.

| Location      | Development time (hrs) | Calibration time (s) | Total time (hrs) | 
|-------------  |------------------------|----------------------|------------------|
| Germany       | 2                      | 2700                 | 02:30:00         |
| Netherlands   | 2                      | 2700                 | 02:30:00         |
| Sweden 1      | 2                      | 2700                 | 02:30:00         |
| Sweden 2      | 2                      | 2700                 | 02:30:00         |

## Additional information

If you want to provide any additional information about your model/submission/results, please do so here.
