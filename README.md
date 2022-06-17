# The Groundwater Modeling Challenge

This repository contains all the information and materials for the Groundwater Time Series Modeling Challenge, [as announced at the 2022 EGU General Assembly](https://doi.org/10.5194/egusphere-egu22-12580). We invite every interested groundwater modeler to model the five different hydraulic head time series found in the data folder, and send in their best possible simulated heads time series. 

Important: Data will be released mid June 2022, if you're interested to join you can already let us know!

**Organisers:** R.A. Collenteur (University of Graz), E. Haaf (Chalmers), T. Liesch & A. Wunsch (KIT), and M. Bakker (TU Delft)

## Background & Objectives
Different types of models can be applied to model groundwater level time series, ranging from purely statistical models (black-box), through lumped conceptual models (grey-box), to process-based models (white-box). Traditionally, physically based, distributed models are predominantly used to solve groundwater problems. In recent years, the use of grey- and black-box models has been receiving increased attention. With this challenge, we want to showcase the diversity of models the can be applied to solve groundwater problems, and systematically asses their differences and performances. 

## Head data
Five hydraulic head time series were selected for this challenge. The monitoring wells are located in sedimentary aquifers, but in different climatological and hydrogeological settings. 

## Model inputs
The model may use (part of) the following data, which is provided as part of the challenge for 2000-2020:

-	Precipitation (daily sums)
-	Temperature (daily means)
-	Potential evaporation (Hamon)

It is permitted to use any other publicly available data (e.g., soil maps) to construct the model. The use of other meteorological data that that provided is not permitted, to ensure that differences between the models are not the result of the meteorological input data. It is also not permitted to use the hydraulic heads as explanatory variables in the model.

## Modeling rules

- Participants may use any type of model.
- The groundwater time series themselves may not be used as model input.
- The hydraulic heads observed between 2000 and 2015, or part thereof, may be used for calibration. 
- The modeling workflow must be reproducible, preferably through the use of scripts, but otherwise described in enough detail to reproduce the results.
- Supplementary model data must be described in sufficient detail and submitted with model outputs

## Model outputs

The model is expected to compute: 
-	The prediction of the hydraulic head at a daily time interval over the entire period 2000-2020, including the 95% prediction interval of the hydraulic head at a daily time interval over the entire period 2000-2020.
-	The step response of 10 mm/day precipitation, and 3 mm/day potential evaporation, including the 95% confidence interval of that step response. Time series to compute this quantity using the calibrated model are provided. (optional)

A form that can be used to submit the results is provided.

## Deliverables

The following data should be submitted:
- Model outputs (see above)
- Model workflow through scripts or description
- Supplementary model data (list and data)
- Estimation of effort (time)

## Model evaluation
The models will be evaluated using several goodness-of-fit metrics and groundwater signatures, computed for both the calibration and the validation period.

## Deadline
The deadline for the challenge is **31/12/2022**. Please make sure to submit before this date. We plan to share the results of this challenge at the EGU General Assembly 2023.

## Submission
Participant can submit their model results as a Pull Request to this Repository, adding a folder with their results in the 'models' folder. The model results must be submitted in a way that they are reproducible, either through the use of scripts (preferred) or detailed description of the modeling process.



## Data



