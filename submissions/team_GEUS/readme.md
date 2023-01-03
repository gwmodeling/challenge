# Team GEUS

In this file the submission is described. 

## Author(s)

- Julian Koch (juko@geus.dk) Geological Survey of Denmark and Greenland, Department of Hydrology 
- Raphael Schneider (rs@geus.dk) Geological Survey of Denmark and Greenland, Department of Hydrology 

## Modeled locations

We modelled the following locations (check modelled locations):

- [x] Netherlands
- [x] Germany
- [X] Sweden 1
- [x] Sweden 2
- [x] USA

## Model description
We applied a Long Short Term Memory (LSTM) model for modelling the gw timeseries. We used off-the-shelf keras functionalities in python to build our model. For each well, we trained models with a single LSTM layer. The mean-squared-error was used to train the central prediction. The confidence intervals (upper and lower boundaries) were trained separately using a quantile regression loss function. The LSTM models were designed to simulate a single day of gw head given a sequence historic meteorological variables.  

## Model workflow to reproduce

1) Interpolate all missing days (between the first and the last observation) in the observed gw head timeseries using linear interpolation. The interpolated timeseries are located in the data folder.    
2) Process additional meteorological variables (based on originally provided data; described in supplementary section below). All additional variables are located in the data folder.
3) Calibrate LSTM parameters: dropout, recurrent_dropout, learning_rate, n_steps (iput sequence length), batchsize, n_cells, as well as the length of the rolling window sums of the supplementary meteorological variables (or the omission of those). For calibration we used the Pareto Archived Dynamically Dimensioned Search (ParaPADDS) in the OSTRICH optimization software. The first four years of each head data served as test data; the remainder as training data.
4) Apply optimized parameters to model the entire period required for the submission (details on LSTM implementation and optimized parameters can be found in the submitted .py files) 
5) The trained LSTM models simulate daily gw head. The timeseries containing all days have also been submitted (*_alldays.csv). Lastly the required days in the submission form has been extracted from the complete daily timeseries. 

## Supplementary model data used

The following variables have been derived from the meteorological data provided: rolling window sums of rainfall rate (180, 365, 730 and 1095 days), rolling window sums of net rainfall rate (180, 365, 730 and 1095 days) and snow storage, snow melt and 90 day rolling window sum of snow melt.   
Snow storage and snow melt is implemented in a very simple manner, with the degree-day method as implemented in the MikeSHE modelling software, with melting temperature: 0 C and max. wet snow fraction: 0 and degree-day melting coefficient: 3 mm/C/d.
All additional variables are located in the data folder.

## Estimation of effort

Please provide an (rough) estimate of the time it took to develop the models (e.g., read in data, pick a model 
structure, etcetera) and calibrate the parameters (if any). If possible, please also state the computational resources that 
were required.

| Location    | Development time (hrs) | Calibration time (hrs) | Total time (hrs) | 
|-------------|------------------------|------------------------|------------------|
| Netherlands | ~ 2                    | 15                     | 17               |
| Germany     | ~ 2                    | 15                     | 17               |
| Sweden 1    | ~ 2                    | 15                     | 17               |
| Sweden 2    | ~ 2                    | 15                     | 17               |
| USA         | ~ 2                    | 15                     | 17               |

## Additional information

Model training and testing has been done on CPU with 20 cores 
