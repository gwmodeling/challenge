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
- Multi-Layer Perceptron (MLP), Rpackage `nnet`, Rosenblatt, F. (1957)
- Random Forest (RF), Breiman, Rpackage `ranger`, L., & Cutler, A. (2001)
- Radial Basis Function support Vector Machine (RBF-SVM), Rpackage `kernlab`, Boser, B. E., Guyon, I. M., & Vapnik, V. N. (1992)
- Polynomial Support Vector Machine (P-SVM), Rpackage `kernlab`, Boser, B. E., Guyon, I. M., & Vapnik, V. N. (1992)

The set of members of the ensemble is different for each of the time series due to the automated tuning and stacking pipline of ensemble candidates. This pipeline is described more in detail below.

The resulting ensembles are as follows (sorry, but apparently there is no rowspan for markdown tables)

| Location    | Ensemble Members | Weights | Penalty | Mixture 
|-------------|------------------------|----------------------|------------------|---------|
| Netherlands | RF                | 0.550   | 0.001         | 0.3  |
|  | RBF-SVM                | 0.161   |        |   |
|  | MLP                | 0.151   |         |   |
|  | RBF-SVM                | 0.141   |        |   |
|  | MLP                | 0.550   |          |   |
| Germany | RF                | 0.615   | 0.001         | 1  |
| | RBF-SVM                | 0.468   |         |   |
| Sweden 1 | RF                | 0.951   | 1e-06         | 0.8  |
|  |     MLP            | 0.251   |       |   |
|  |     RBF-SVM            | 0.0300   |         |   |
| Sweden 2 | RF                | 0.402   | 0.1         | 0  |
|  |     RBF-SVM            | 0.286   |         |   |
|  |     RBF-SVM            | 0.253   |         |   |
|  |     MLP            | 0.229   |       |   |
| USA | RF                | 0.754   | 1e-06         | 0.6  |
|  |     RBF-SVM            | 0.244   |         |   |
|  |     RBF-SVM            | 0.109   |         |   |


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

The most recent 10% of each time series was used for testing. The remaining 90% were used for tuning the hyperparameters. A 10-fold time series crossvalidation was used as resampling strategy as I used non-sequential learners only. The tuning was conducted with a simple grid search with 5 different values/levels for each parameter, leading to $n_{levels}^{n_{hyperparameters}}$ combinations. The tuning was done for all learners for all four feature engineering recipes and for all locations. RMSE was used as metric during tuning. The two best performing models of each learner were chosen as possible candidates for building the ensemble. Once the performance was known, this pipeline was applied to the fulll training period defined by the organizers.

The code can be found in this github release: [https://github.com/MxNl/GroundwaterModellingChallenge/releases/tag/v1.0.0](https://github.com/MxNl/GroundwaterModellingChallenge/releases/tag/v1.0.0)

## Supplementary model data used

No additional data was used for modelling. Feature engineering is only based on the provided data.

## Estimation of effort

The provided development time also includes some trial and error iterations with different models and hyperparameter ranges, bug fixing etc.
The tuning for all locations took around 11 hours and is not included in the table. 

| Location    | Development time (hrs) | Calibration time (s) | Total time (hrs) | 
|-------------|------------------------|----------------------|------------------|
| all together | ~  20                 | 8s - 90 (depending on location)   | ~ 20:02:30         |

## Additional information

Just some general remarks:
- No ChatGPT was used to produce this markdown file :smiley:.
- I chose this setup of learners out of curiosity, not because I personally think they would compete with sequential or recurrent models like CNN or LSTM.
- Overfitting is still an issue for some of the locations. Nested crossvalidation for tuning would have been nice to reduce overfitting, but wasn't implemented due to lack of time :cry:.
