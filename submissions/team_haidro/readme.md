# Team Example

In this file the submission is described. 

## Author(s)

- Tim Franken (Sumaqua)

## Modeled locations

We modelled the following locations (check modelled locations):

- [X] Netherlands
- [X] Germany
- [X] Sweden 1
- [X] Sweden 2
- [X] USA

## Model description

We used a multi-timescale LSTM similar to what is described in Franken et. al (2022) which was based on the work from
Martin Gauch et. al. For the uncertainty estimation we used monte-carlo dropout procedure in inference. The model is
implemented in Tensorflow (Keras). A single LSTM model was trained for each location.

References:
https://meetingorganizer.copernicus.org/EGU22/EGU22-6263.html
https://gauchm.github.io/

## Model workflow to reproduce

1. First estimate the optimal hyperparameters using the script optimize_params.py. This should work immediately on the
data and provide the optimized hyperparameters
2. Train the final models and prepare the submission using the script prepare_submission

## Supplementary model data used

No additional information was obtained and/or used.

## Estimation of effort

Please provide an (rough) estimate of the time it took to develop the models (e.g., read in data, pick a model 
structure, etcetera) and calibrate the parameters (if any). If possible, please also state the computational resources
that were required.

| Location    | Development time (hrs) | Calibration time (hrs) | Total time (hrs) | 
|-------------|------------------------|----------------------  |------------------|
| Netherlands | ~ 2                    | ~ 2                    | 04:00:00         |
| Germany     | ~ 2                    | ~ 2                    | 04:00:00         |
| Sweden 1    | ~ 2                    | ~ 2                    | 04:00:00         |
| Sweden 2    | ~ 2                    | ~ 2                    | 04:00:00         |
| USA         | ~ 2                    | ~ 2                    | 04:00:40         |

## Additional information

First, thanks a lot for organizing this challenge and good luck processing all the submissions!

Secondly, this is certainly not the best possible submission using a LSTM model. Due to time constraints I've not
been able to fully investigate the architectural options, do a proper hyperparmeter tuning and / or more advanced
uncertainty estimation. On a similar note also the code could certainly use some refactoring and additional documentation.
I was however to curious to see how this model would compare against other state of the art models not to submit is.
