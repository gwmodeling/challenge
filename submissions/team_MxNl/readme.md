# Team MxNl
## Author(s)

- Maximilian Nölscher (German Federal Institute for Geoscience and Resources (BGR)): [ORCID](https://orcid.org/0000-0001-5606-1900)

## Modelled locations

We modelled the following locations:

- [x] Netherlands
- [x] Germany
- [x] Sweden 1
- [x] Sweden 2
- [x] USA

## Model description

I used an ensemble of different shallow, non-sequential learners 
- Multi-Layer Perceptron (MLP), 
- Boosting Trees (BT), 
- Random Forest (RF), 
- Radial Basis Function support Vector Machine (RBF-SVM), 
- Polynomial Support Vector Machine (P-SVM)). 

The set of members of the ensemble is different for each of the time series due to the automated tuning and stacking pipline of ensemble candidates. This pipeline is described more in detail below. the model XX as described in detail in XX et al. (1979).
implemented in the XX software package that was used here.

## Model workflow to reproduce

### Environment
- R Version: 4.1.2
- Package Versions: Can be restored by using the package `renv`:
1. Install `renv`
```
install.packages("renv")
```
2. Install required packages with correct versions
```
renv::restore()
```
The modelling workflow was implemented using the metapackage and machine learning framework `tidymodels` within a `targets` pipeline.

### Modelling Workflow
The modelling workflow starts with data preparation, such as the aggregation of predictors interval to interval of groundwater levels and imputation of missing values. Secondly, feature engineering was conducted. Four different feature engineering recipes were used for training the different learners. They all share the following step: Three additional predictors were added for each predictor consisting of aggregated values for the previous week, month and 3 months for each time step. 
From this point, the four feature engineering recipes differ. For details on differences of these recipes, I refer to the linked source code. In general, the following steps are used:
- Log transformation of et and rr
- Removal of highly correlated predictors
- Conversion of numeric predictors into principle components
- Standardization of numeric predictors
- Lagged predictors were added for the predictors rr (precipitation) and et (evapotranspiration) for up to 25 weeks.

The most recent 10% of each time series was used for testing. The remaining 90% were used for tuning the hyperparameters. A 5-fold crossvalidation with ?? repetitions was used as resampling strategy as I used non-sequential learners only. The tuning was conducted with a simple grid search with ?? different values for each parameter, leading to $n_{levels}^{n_{hyperparameters}}$ combinations


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

Just some general remarks:
- No ChatGPT was used to produce this markdown file. 
- Nested crossvalidation for tuning would have been nice to reduce overfitting, but wasn't implemented due to lack of time.
