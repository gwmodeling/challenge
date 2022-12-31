# Team TUD

In this file the submission is described. 

## Author(s)

- Max Rudolph (Institute of Groundwater Management, TU Dresden, Germany)
- Alireza Kavousi (Institute of Groundwater Management, TU Dresden, Germany)

## Modeled locations

We modelled the following locations (check modelled locations):

- [x] Netherlands
- [ ] Germany
- [ ] Sweden 1
- [ ] Sweden 2
- [x] USA

## Model description

We followed a modelling approach described in detail [here](https://www.tensorflow.org/tutorials/structured_data/time_series) (TensorFlow Developers, 2022). The model is an artificial neural network (ANN) type of model, and a long short-term memory (LSTM) model, more specifically. The model is implemented in the `TensorFlow (v 2.8.0)` software package (TensorFlow Developers, 2022) that was used here. Starting from the single-output LSTM approach described in [this example](https://www.tensorflow.org/tutorials/structured_data/time_series), we adapted the approach to not include the target data, i.e., measured heads.
We selected relevant features, or system input data, by simply assessing the (linear) covariance structure between features and target. Features as well as target were normalized to the range from zero to unity. The model was trained using batches of data windows, where a number of past input data was used to predict the target for the current time step. The number of past steps to include as well as the model architecture was obtained from a trial-and-error procedure, where the linear correlation coefficient between observed and simulated values was tried to be minimized for both of the modelled cases simulataneously. To enable an assessment of simulation uncertainty, the model was trained a total of 100 times, including a (random) 10% drop-out before the first LSTM layer. This resulted in an ensemble of simulations, from which a simulation mean as well as confidence intervals were calculated. The model structure was developed to be universally applicable, where only the relevant input features need to be selected for a new case.

## Model workflow to reproduce

The model was set up in `Jupyter Notebooks`, one for each modelled case (see `Model_Netherlands.ipynb` and `Model_USA.ipynb`). The environment used can be created from the `environment_TUD.yml`.

## Supplementary model data used

No additional information was obtained and/or used.

## Estimation of effort

We calibrated the models on a personal computer with limited computational power. It is assumed that the calibration time can be substantially reduced, if high-performance-computing resources are available. Initially, the model was set up for the Netherlands case. The application to a new case was straightforward, so the development time was shortened.
Because of the limited computational resources available, we were not able to apply the approach to all available cases. We will, however, still apply our model to the remaining cases and supply the results, even if they will not be formally considered subsequently. 

| Location    | Development time (hrs) | Calibration time (d) | Total time (d) | 
|-------------|------------------------|----------------------|------------------|
| Netherlands | ~ 4                    | ~ 2                  | ~ 2              |
| Germany     |                        |                      |                  |
| Sweden 1    |                        |                      |                  |
| Sweden 2    |                        |                      |                  |
| USA         | ~ 0.5                  | ~ 2                  | ~ 2              |

## Additional information

No additional information is provided.