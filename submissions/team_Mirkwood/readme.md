# Team Mirkwood

In this file the submission is described. 

## Author(s)

- Antoine Di Ciacca (Environmental Research, Lincoln Agritech Ltd, Lincoln, New Zealand)


## Modeled locations

We modelled the following locations (check modelled locations):

- [x] Netherlands
- [x] Germany
- [x] Sweden 1
- [x] Sweden 2
- [x] USA

## Model description

We used an ensemble of random forest models implemented in the R language using the 'tidymodels' framework (Kuhn and Wickham, 2020) 
and the 'ranger' implementation of random forest (Wright and Ziegler, 2017). This approach is derived from the one described in Di Ciacca et al, 2022
but with several modifications to fit the purpose of this challenge.

#References

Kuhn, M. and Wickham, H.: Tidymodels: a collection of packages for modeling and machine learning using tidyverse principles., https:
//www.tidymodels.org, 2020.

Wright, M. N. and Ziegler, 670 A.: ranger: A Fast Implementation of Random Forests for High Dimensional Data in C++ and R, Journal of
Statistical Software, 77, 1–17, https://doi.org/10.18637/jss.v077.i01, 2017.

Di Ciacca, A., Wilson, S., Kang, J., and Wöhling, T.: Deriving transmission losses in ephemeral rivers using satellite imagery and machine learning, 
EGUsphere [preprint], https://doi.org/10.5194/egusphere-2022-833, 2022. 

## Model workflow to reproduce

The model workflow can be found in the commented scripts attached. They were run using R version 4.2.1.

## Supplementary model data used

No additional information was obtained and/or used.

## Estimation of effort


| Location    | Development time (hrs) | Calibration time (s)                | Total time (hrs) | 
|-------------|------------------------|-----------------------------------  |------------------|
| Netherlands | ~ 3                    | 170 + 5400 for uncertainty analysis | 4.5              |
| Germany     | ~ 3                    | 150 + 4680 for uncertainty analysis | 4.3              |
| Sweden 1    | ~ 3                    | 20 + 1800 for uncertainty analysis  | 3.5              |
| Sweden 2    | ~ 3                    | 20 + 1800 for uncertainty analysis  | 3.5              |
| USA         | ~ 3                    | 80 + 1800 for uncertainty analysis  | 3.5              |

All calculations were done with a destktop computer:
Processor = Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz   2.90 GHz
RAM = 32.0 GB (31.8 GB usable)

## Additional information

Same method applied to all sites. 
I don't think the uncertainty analysis worked very well but I hope that comparing this simple and flexible model with more advanced approaches can be interesting.
Thanks for organising this challenge!
Antoine