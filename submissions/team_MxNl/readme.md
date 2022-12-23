# Team MxNl

In this file the submission is described. 

## Author(s)

- Maximilian Nölscher (German Federal Institute for Geoscience and Resources (BGR)) [ORCID](https://orcid.org/0000-0001-5606-1900)

## Modeled locations

We modelled the following locations (check modelled locations):

- [x] Netherlands
- [x] Germany
- [x] Sweden 1
- [x] Sweden 2
- [x] USA

## Model description

I used an ensemble of different shallow, non-sequential learners (Multi-Layer Perceptron (MLP), Boosting Trees (BT), Radial Basis Function support Vector Machine (RBF-SVM), Polynomial Support Vector Machine (P-SVM)). The set of members of the ensemble is different for each of the time series due to the automated tuning and stacking pipline of ensemble candidates. This pipeline is described more in detail below. the model XX as described in detail in XX et al. (1979).
implemented in the XX software package that was used here.

## Model workflow to reproduce

Please provide a detailed description of the modeling workflow here, in such a way that the results may be 
reproduced independently by others. The preferred way to ensure reproducibility is to provide a commented script and 
environment settings.

## Supplementary model data used

No additional data was used for modelling. Feature engineering is only based on the provided data.

## Estimation of effort

Please provide an (rough) estimate of the time it took to develop the models (e.g., read in data, pick a model 
structure, etcetera) and calibrate the parameters (if any). If possible, please also state the computational resources that 
were required.

| Location    | Development time (hrs) | Calibration time (s) | Total time (hrs) | 
|-------------|------------------------|----------------------|------------------|
| Netherlands | ~ 2                    | 40                   | 02:00:40         |
| Germany     |                        |                      |                  |
| Sweden 1    |                        |                      |                  |
| Sweden 2    |                        |                      |                  |
| USA         |                        |                      |                  |

## Additional information

If you want to provide any additional information about your model/submission/results, please do so here.
