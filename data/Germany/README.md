# Monitoring well Germany

## Location description
The well is situated in Bavaria, in the Upper Jurassic Malm Karst aquifer. It is a deep, confined aquifer (partly artisian), which is overlain by a local alluviual aquifer in a small river valley. Surface elevation is about 375 masl, depth to groundwater 0.9 m on average.

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

- **Training period:** 2002-05-01 to 2016-12-31
- **Testing period:** 2017-01-01 to 2021-12-31

## GitHub Discussion about this data

If you have any questions or comments about the data provided for this location, please post in the [GitHub 
Discussion related to this well](https://github.com/gwmodeling/challenge/discussions/4). This way, all participants 
will have the same information.

## Data Acknowledgments

- We acknowledge the E-OBS dataset from the EU-FP6 project UERRA (https://www.uerra.eu) and the Copernicus Climate 
Change Service, and the data providers in the ECA&D project (https://www.ecad.eu)
