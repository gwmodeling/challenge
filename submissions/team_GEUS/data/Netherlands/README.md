# Monitoring well Netherlands

## Location description
This monitoring well is located in the north of the Netherlands in the province of Drenthe. The surface elevation is 
about 11.35 meters above mean sea level, and the well is screened from -0.05 to -0.95 below surface level. The well 
is located in an unconfined aquifer, consisting of a ~1.5 meter top-layer of peat materials underlain by fine sands. 
The area is characterized by many small surface water drainages, through which groundwater is likely drained above a 
certain groundwater level.

![Head data](head_data.png)

## Input data description

The following input data are provided to model the head time series. This data were collected from the E-OBS dataset 
v25.0e at 0.1deg grid size.

- Daily Precipitation (RR) in mm/d.
- Daily mean temperature (TG) in degree Celsius.
- Daily minimum temperature (TM) in degree Celsius.
- Daily maximum temperature (TX) in degree Celsius.
- Daily averaged sea level pressure (PP) in hPa.
- Daily averaged relative humidity (HU) in %.
- Daily mean wind speed (FG) in m/s.
- Daily mean global radiation (QQ) in W/m2.
- Potential evaporation (ET) computed with Makkink in mm/d.

## Calibration and testing data

The head data are split into a training and testing period. Data are provided for the training/calibration period. Participants have to provide a simulated time 
series for the entire period:

- **Training period:** 2000-01-01 to 2015-09-10
- **Testing period:** 2016-01-01 to 2021-12-31

## GitHub Discussion about this data

If you have any questions or comments about the data provided for this location, please post in the [GitHub 
Discussion related to this well](https://github.com/gwmodeling/challenge/discussions/5). This way, all participants 
will have the same information.

## Data Acknowledgments

- We acknowledge the E-OBS dataset from the EU-FP6 project UERRA (https://www.uerra.eu) and the Copernicus Climate 
Change Service, and the data providers in the ECA&D project (https://www.ecad.eu)
- We acknowledge DINOloket for providing the hydraulic head time series (https://www.dinoloket.nl/).  
