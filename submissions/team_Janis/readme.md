# Team Janis

In this file the submission is described. 

## Author(s)

- Jānis Bikše (University of Latvia, Faculty of Geography and Earth Sciences)

## Modeled locations

We modelled the following locations (check modelled locations):

- [x] Netherlands
- [x] Germany
- [x] Sweden 1
- [x] Sweden 2
- [x] USA

## Model description

Random forest model. It was implemented in R (version 4.2.2) using *ranger* package (version 0.14.1) within *tidymodels* (version 1.0.0) environment. 


## Model workflow to reproduce

The model can be reproduced by provided R code (*RF_model_Janis.R*)

## Supplementary model data used

No additional data was used. 

## Estimation of effort

Please provide an (rough) estimate of the time it took to develop the models (e.g., read in data, pick a model 
structure, etcetera) and calibrate the parameters (if any). If possible, please also state the computational resources that 
were required.

| Location    | Development time (hrs) | Calibration time (s) | Total time (hrs) | 
|-------------|------------------------|----------------------|------------------|
| Netherlands | ~ 0.5                  | 345                  | 00:45:45         |
| Germany     | ~ 5                    | 310                  | 05:05:10         |
| Sweden 1    | ~ 0.5                  | 48                   | 00:30:48         |
| Sweden 2    | ~ 0.5                  | 60                   | 00:31:00         |
| USA         | ~ 1                    | 289                  | 01:04:49         |

Most (roughtly estimated) development time was spent to create a general structure of the workflow (using Germany data) and partly to prepare new variables (including the assessment of their importance), while less time was necessary to scale this approach to other case studies. Eventually models for all locations were merged into a single script *RF_model_Janis.R*. Calibration times correspond to time spent on tuning *mtry* parameter (7 mtry values) with resamples (8 splits of resamples) for Random forest model, which was run on 8 parallel CPUs

## Additional information

The model was run on a laptop equipped with AMD Ryzen 7 Pro 4750U CPU. 