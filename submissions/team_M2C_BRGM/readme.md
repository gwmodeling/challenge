# Team M2C CNRS & BRGM

In this file the submission is described. 

## Author(s)

- Sivarama Krishna Reddy Chidepudi (M2C CNRS & BRGM) sivaramakrishnareddy.chidepudi@univ-rouen.fr
- Abel Henriot (BRGM) a.henriot@brgm.fr
- Nicolas Massei(M2C CNRS) nicolas.massei@univ-rouen.fr
- Abderrahim Jardani(M2C CNRS) abderrahim.jardani@univ-rouen.fr

## Modeled locations

We modelled the following locations :

- [x] Netherlands
- [X] Germany
- [X] Sweden 1
- [X] Sweden 2
- [X] USA

## Model description

We used the Boundry corrected maximal overlap wavelet transfrom deep learning model (BC-MODWT-DL) models for four wells and BILSTM for one well as described in detail in Chidepudi et al. (2022). This is a deep learning model with waveleth preprocessing. The model is 
implemented python and script is given in the script repository.

## Model workflow to reproduce
Scripts necessary to reproduce the work is submitted in the scripts folder

The extra library wmtsa_cima is needed, and the weel is given in the wmtsa_cima repository. Installation can be done folowing instruction in the readme.md file inside that folder.

Please provide a detailed description of the modeling workflow here, in such a way that the results may be 
reproduced independently by others. The preferred way to ensure reproducibility is to provide a commented script and 
environment settings.

## Supplementary model data used

'No additional information was obtained and/or used.'

## Estimation of effort

Please provide an (rough) estimate of the time it took to develop the models (e.g., read in data, pick a model 
structure, etcetera) and calibrate the parameters (if any). If possible, please also state the computational resources that 
were required.

| Location    | Development time (hrs) | Calibration time (s) | Total time (hrs) | 
|-------------|------------------------|----------------------|------------------|
| Netherlands | ~ 2                    | 40                   | 02:00:40         |
| Germany     |                        |                      |                  |
| Sweden 1    |                        |                      |                  |
| Sweden 2    |                        |                      |                  |
| USA         |                        |                      |                  |

## Additional information
All this work was conducted in Python version 3.8.13, using a
Dell workstation with an NVIDIA Quadro RTX 5000 GPU and 128GB RAM


Retained models :
USA, Germany, Netherlands : LSTM + WT
Swenden 1 : BILSTM
Sweden 2 : GRU + WT

# word on how we selected the best model :
