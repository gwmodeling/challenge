# Team TUV

In this file the submission is described. 

## Authors

-Anna Pölz (TU Wien)

-Ali Obeid (TU Wien) 

-Ahmad Ameen (TU Wien)

## Modeled locations

We modelled the following locations (check modelled locations):

- [x] Netherlands
- [x] Germany
- [X] Sweden 1
- [x] Sweden 2
- [x] USA

## Model description

We used an encoder-only version of the Transformer model as described in detail in Vaswani et al. (2017). Since it was applied for time series forecasting we omitted the input embedding. This is a deep learning type of model. The model is implemented in python. Tensorflow and Keras software packages were used here.

## Model workflow to reproduce

Normalization was used as a preprocessing step. All independent variables were used for calibrating the model. Input windows of 30 days were chosen as input for the model to predict the hydraulic head. The models were tuned using keras tuner (Random search) with maximal trials of 10. The following variables were tuned: head size, filter size and MLP units. The further parameters were set to the following values: number of heads = 1, number of encoder blocks =2, dropout=0.1, MLP dropout=0.1. The Transformer model was implemented as an encoder-only version. To provide prediction intervals Monte Carlo dropout was used and the intervals were calibrated for 95% percentage of coverage (prediction interval covers measured value) on the calibration set. The environment is saved under GW_challenge.yml. 

## Supplementary model data used

No additional information was obtained and/or used.

## Estimation of effort

Please provide an (rough) estimate of the time it took to develop the models (e.g., read in data, pick a model 
structure, etcetera) and calibrate the parameters (if any). If possible, please also state the computational resources that 
were required.

| Location    | Development time (hrs) | Calibration time (s) | Total time (hrs) | 
|-------------|------------------------|----------------------|------------------|
| Netherlands |  2h                    | ~600                 | 02:10:00         |
| Germany     |  6h                    | ~600                 | 06:10:00         |
| Sweden 1    |  3h                    | ~300                 | 03:05:00         |
| Sweden 2    |  2h                    | ~300                 | 02:05:00         |
| USA         |  2h                    | ~600                 | 02:10:00         |

