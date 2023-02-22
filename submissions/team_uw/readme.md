# Team Example

In this file the submission is described. 

## Author(s)

- Morteza Behbooei (University of Waterloo, ON, Canada)
- Jimmy Lin (University of Waterloo, ON, Canada)
- Bryan Tolson (University of Waterloo, ON, Canada)
- Rojin Meysami (University of Waterloo, ON, Canada)

## Modeled locations

We modelled the following locations (check modelled locations):

- [x] Netherlands
- [x] Germany
- [ ] Sweden 1
- [ ] Sweden 2
- [ ] USA

## Model description

This is a RNN model based on LSTM blockes. (many to one configuration) We used this model because of the sequential nature of the data. The model has been implemented by pytorch package. As noticed in the challlenge description we didn't used the heads as input for the model (however by using heads as input for training model gets much accurate). For the training step, we used previous 90 days window of input data for every target. Also, hyperparameters have been tuned through a grid search method (model_pytorch_tuning.ipynb). By using Cuda based pytorch training, the training time for these models is under 30 seconds. The outputs for first 90 days are repetitive because of the length of RNN model.(90 days)

## Model workflow to reproduce

You can use the model_pytorch_load.ipynb file for reproducing the outputs. In the learning state we saved the trained models (Germany.pth, Netherlands.pth) and the model_pytorch_load.ipynb file loads these files and creates the outputs. You should just set the "country" parameter to your prefered country.
There is a folder named "Reliability" that you can use to see the code for reliability analysis of the model. We trained the model 1000 times for each dataset and added the results of 95 percent upper and lower bounds to the results. All the generated models are available in the zip file with the dataset name in this folder. You should extract the zip file to run the reliability code again.

## Supplementary model data used

No supplementary model data was used.

## Estimation of effort

We used a personal computer with 3070 Nvidia GPU. We used GPU for training so the training times are very small.

| Location    | Development time (days) | Calibration time (hours) | Total time (days) | 
|-------------|------------------------|----------------------|------------------|
| Netherlands | ~ 3                    | 5                   | 3         |
| Germany     | ~ 3                       | 5                     |  3                |
| Sweden 1    |                        |                      |                  |
| Sweden 2    |                        |                      |                  |
| USA         |                        |                      |                  |

## Additional information

You can see an example of accuracy when we add heads to our input data in the model.ipynb file.