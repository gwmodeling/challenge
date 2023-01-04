# Team RouhaniEtAl

In this file the submission is described. 

## Author(s)

- Amir Rouhani (Department of Environment, Land and Infrastructure Engineering, Politecnico di Torino, Turin, Italy)
- Jaime Gómez-Hernández (Universitat Politècnica de València)
- Seifeddine Jomaa (Department of Aquatic Ecosystem Analysis and Management, Helmholtz Centre for Environmental Research - UFZ, Magdeburg, Germany (seifeddine.jomaa@ufz.de))

## Modeled locations

We modelled the following locations (check modelled locations):

- [x] Netherlands
- [x] Germany
- [X] Sweden 1
- [x] Sweden 2
- [x] USA

## Model description

"We used Convolutional Neural Networks (CNNs) as described in detail in Wunsch, A et al. (2022). This is a deep learning type of model. The model is 
implemented in the Python TensorFlow, and its Keras API. Further, the following libraries were used: Numpy, Pandas, Scikit-Learn, BayesOpt, Matplotlib, Unumpy, and SHAP."

## Model workflow to reproduce

"The CNNs used in this study comprise a 1D convolutional layer with fixed kernel size (three) and optimized number of filters, followed by a Max-Pooling layer and a 
Monte-Carlo dropout layer, applying a fixed dropout of 50% to prevent the model from overfitting. This dropout rate is quite high and forces the model to perform a
very robust training. A dense layer with an optimized number of neurons follows, followed by a single output neuron. We used the Adam optimizer for a maximum of 100
training epochs with an initial learning rate of 0.001 and applied gradient clipping to prevent exploding gradients. Early stopping with patience of 15 epochs was 
applied as another regularization technique to prevent the model from overfitting the training data. Several model hyperparameters were optimized using Bayesian
optimization: training batch size (16–256); input sequence length (1–365 for daily datasets) & (1-52 for weekly datasets); number of filters in the 1D-Conv 
layer (1–256); size of the first dense layer (1–256)."

## Supplementary model data used

No additional information was obtained and/or used.

## Estimation of effort

Please provide a (rough) estimate of the time it took to develop the models (e.g., read in data, pick a model 
structure, etcetera) and calibrate the parameters (if any). If possible, please also state the computational resources that 
were required.

Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz   2.00 GHz processor
8.00 GB RAM 

| Location    |   Training time (hrs)  | Calibration time (s) | Total time (hrs) | 
|-------------|------------------------|----------------------|------------------|
| Netherlands | ~ 3                    |                      | 03:00:00         |
| Germany     | ~ 3                    |                      | 03:00:00         |
| Sweden 1    | ~ 1                    |                      | 01:00:00         |
| Sweden 2    | ~ 1                    |                      | 01:00:00         |
| USA         | ~ 3                    |                      | 03:00:00         |

## Additional information

If you want to provide any additional information about your model/submission/results, please do so here.