# Team Example

In this file the submission is described. 

## Author(s)

- Morteza Behbooei (University of Waterloo, ON, Canada)
- Jimmy Lin (University of Waterloo, ON, Canada)
- Rojin Meysami (University of Waterloo, ON, Canada)

## Modeled locations

We modelled the following locations (check modelled locations):

- [x] Netherlands
- [x] Germany
- [ ] Sweden 1
- [ ] Sweden 2
- [ ] USA

## Model description

This is a RNN model based on LSTM blockes. (many to one configuration) We used this model because of the sequential nature of the data. The model has been implemented by pytorch package. As noticed in the challlenge description we didn't used the heads as input for the model (however by using heads as input for training model gets much accurate). For the training step, we used previous 90 days window of input data for every target. Also, hyperparameters have been tuned through a grid search method (model_pytorch_tuning.ipynb). By using Cuda based pytorch training, the training time for these models is under 30 seconds.

## Model workflow to reproduce

You can use the model_pytorch_load.ipynb file for reproducing the outputs. In the learning state we saved the trained models (Germany.pth, Netherlands.pth) and the model_pytorch_load.ipynb file loads these files and creates the outputs. You should just set the "country" parameter to your prefered country.

## Supplementary model data used

No supplementary model data was used.

## Estimation of effort

We used a personal computer with 3070 Nvidia GPU. We used GPU for training so the times are very small.

| Location    | Development time (s) | Calibration time (s) | Total time (s) | 
|-------------|------------------------|----------------------|------------------|
| Netherlands | ~ 30                    |                    | 30         |
| Germany     | ~ 30                       |                      |  30                |
| Sweden 1    |                        |                      |                  |
| Sweden 2    |                        |                      |                  |
| USA         |                        |                      |                  |

## Additional information

You can see an example of accuracy when we add heads to our input data in the model.ipynb file.