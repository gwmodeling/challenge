# The Groundwater Modeling Challenge

**Update 2022/09/19: Data is released and the challenge has started !**

This repository contains all the information and materials for the Groundwater Time Series Modeling Challenge, [as 
announced at the 2022 EGU General Assembly](https://doi.org/10.5194/egusphere-egu22-12580). We invite every 
interested groundwater modeler to model the five different hydraulic head time series found in the data folder, and 
send in their best possible simulated head time series. 

**Organisers:** R.A. Collenteur (Eawag), E. Haaf (Chalmers), T. Liesch & A. Wunsch (KIT), and M. Bakker 
(TU Delft)

## Background & Objectives

Different types of models can be applied to model groundwater level time series, ranging from purely statistical 
models (black-box), through lumped conceptual models (grey-box), to process-based models (white-box). Traditionally, 
physically based, distributed models are predominantly used to solve groundwater problems. In recent years, the use 
of grey- and black-box models has been receiving increased attention. With this challenge, we want to showcase the 
diversity of models that can be applied to solve groundwater problems, and systematically asses their differences and 
performances.

## Input and hydraulic head data

Five hydraulic head time series were selected for this challenge. The monitoring wells are located in sedimentary 
aquifers, but in different climatological and hydrogeological settings. Depending on the location. different input 
time series are available to model the heads. Please find all data and descriptions in the [data folder](https://github.com/gwmodeling/challenge/tree/main/data).

It is permitted to use any other publicly available data (e.g., soil maps) to construct the model. The use of other 
meteorological data that that provided is not permitted, to ensure that differences between the models are not the 
result of the meteorological input data. It is also not permitted to use the hydraulic heads as explanatory 
variables in the model.

## Modeling rules

- Participants may use any type of model.
- The groundwater time series themselves may not be used as model input.
- The modeling workflow must be reproducible, preferably through the use of scripts, but otherwise described in 
  enough detail to reproduce the results.
- Supplementary model data must be described in sufficient detail and submitted with model outputs.

## Model outputs and deliverables

The model is expected to compute: 

-	The prediction of the hydraulic head for the dates found in the submission files in the  'team_example' folder, 
    including the 95% prediction interval of the hydraulic head at a daily time interval over the entire 
     calibration and validation period (see data folders for specific periods for each location).

Forms that can be used to submit the results are provided in the [submissions folder](https://github.com/gwmodeling/challenge/tree/main/submissions). 
There you can also find more detailed on what to submit.

## Model evaluation

The models will be evaluated using several goodness-of-fit metrics and groundwater signatures, computed for both the 
calibration and the validation period. The data for the validation period is not make public yet and will be 
released after the challenge ended.

## Deadline

The deadline for the challenge is **31/12/2022. Late submission are allowed untill 5th of January 24:00 CET.** Please make sure to submit before this date. We plan to share the results of this challenge at the EGU General Assembly 2023.

## Participation & Submission
If you intend to participate, [please open a GitHub Issue for your team](https://github.com/gwmodeling/challenge/issues), such that we can track the participating teams.

Participant can submit their model results as a Pull Request to this Repository, adding a folder with their results 
in the 'submissions' folder. The model results must be submitted in a way that they are reproducible, either through 
the use of scripts (preferred) or detailed description of the modeling process. See the [submissions folder](https://github.com/gwmodeling/challenge/tree/main/submissions) for a more detailed description on how and what to submit.

After the challenge we intend to write an article to submit to a peer-reviewed journal with all the organisers and participants.

## Questions/ Comments ?

To make sure everyone has access to the same information we ask you to put any questions that are of general 
interest to all participants in the [GitHub Discussion forum](https://github.com/gwmodeling/challenge/discussions).


