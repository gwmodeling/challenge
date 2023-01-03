# Team M2C CNRS & BRGM

In this file the submission from our team is described. 

## Author(s)

- Sivarama Krishna Reddy Chidepudi (M2C CNRS & BRGM) sivaramakrishnareddy.chidepudi@univ-rouen.fr
- Abel Henriot (BRGM) a.henriot@brgm.fr
- Nicolas Massei(M2C CNRS) nicolas.massei@univ-rouen.fr
- Abderrahim Jardani(M2C CNRS) abderrahim.jardani@univ-rouen.fr

## Modelled locations

We modelled the following locations :

- [x] Netherlands
- [X] Germany
- [X] Sweden 1
- [X] Sweden 2
- [X] USA

## Model description

We used the Boundary corrected-maximal overlap wavelet transform deep learning (BC-MODWT-DL) models for four wells (Germany,Netherlands,Sweden_2,USA) and Bidirectional LSTM (BiLSTM) for one well(Sweden_1) using the approach as described in detail in Chidepudi et al. (2022). These are deep learning models with wavelet preprocessing.  

## Model workflow to reproduce
All the models are implemented in python and neccesary scripts to reproduce the work  are given in the scripts folder. We used tensorflow library for DL models and OPTUNA for hyperparameter tuning using bayesian optimisation. For data handling we used pandas and numpy libraries. The extra library wmtsa_cima is needed for the implementation of MODWT, and the wheel is given in the wmtsa_cima repository. Installation can be done folowing instruction in the readme.md file inside the scripts/mwtsa_cima folder. 

Scripts provided are self-explanatory. The results can be reproduced by changing just the well name and working directory in the beginning of the scripts.We provided two scipts one with BC-MODWT-DL (DL_MODWT_4Wells.py) and other one is for simple DL models(Sweden_1_script_BLSTM.py).   

## Supplementary model data used

'No additional information was obtained and/or used.'

## Estimation of effort


Total development is split over the wells and even though same script can be used for all wells, it took time to check for multiple possibilties and the select one suitable combination for each well 

| Location    | Development time (hrs) | Calibration time (Mins) | Total time (hrs) | 
|-------------|------------------------|----------------------|------------------|
| Netherlands | ~5                     | 30                   | 5:30:00          |
| Germany     | ~5                     | 30                   | 5:30:00          |
| Sweden 1    | ~5                     | 30                   | 5:30:00          |
| Sweden 2    | ~5                     | 30                   | 5:30:00          |
| USA         | ~5                     | 30                   | 5:30:00          |

## Additional information
All this work was conducted in Python version 3.8.13, using a
Dell workstation with an NVIDIA Quadro RTX 5000 GPU and 128GB RAM


Final subimitted simulations for each well were using the following combinations  :
USA, Germany, Netherlands : LSTM + BC-MODWT
Swenden 1 : BILSTM
Sweden 2 : GRU + BC-MODWT

Also all the models used here are stacked models with multiple layers with layers and other hyperparemeters being optimised for each well.
 
You might notice that out start date of simulations is different from 2002-01-01 as we had to remove boundary affected coefficients hence our simulations start at later date.

# how we selected the above combinations :
We ran the simulations for each well for all possible combinations i.e., 3 deep learning models (GRU,LSTM,BiLSTM) with and without BC-MODWT (also for different wavelets la8 to la16) pre-processing and validated this models on last 20% of training set for different metrics (RMSE,MAE,R2) and then choose one suitable model for each well even though in most of the cases similar results were obtained, We choose combination that requires removing of less boundary-affected coefficients. The necessity of removing these coefficients and other details can be seen from Chidepudi et al (2022). All the combinations can be checked using the scripts provided by changing the modtype (LSTM,BiLSTM,GRU) and Wavelet types (La8,La10....La16) 

For more details, please contact us using the emails provided in the beginning. We are happy to hear from you and contribute further in the challenge.


## References
Chidepudi, S. K. R., Massei, N., Jardani, A., Henriot, A., Allier, D., & Baulon, L. (2022). A wavelet-assisted deep learning approach for simulating groundwater levels affected by low-frequency variability. Science of The Total Environment, 161035. https://doi.org/10.1016/j.scitotenv.2022.161035
