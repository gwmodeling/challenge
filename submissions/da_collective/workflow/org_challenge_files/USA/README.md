# Monitoring well USA

## Location description
The well is located in the state of Connecticut in the USA. The well is screened in a confined bedrock aquifer. The 
aquifer consists of crystalline noncarbonated rock (predominantly metamorphic schist and gneiss that are highly 
folded with numerous fractures and joints). Surface elevation is at approximately 156.9 m; the well screen is 
approximately 135.3 m below surface level. The distance to the nearby surface water is approximately 6.83 km.

![Head data](head_data.png)

## Input data description

The following input data are provided to model the head time series. 

- Daily Precipitation (PRCP) in mm/d.
- Daily minimum temperature (TMin) in degree Celsius.
- Daily maximum temperature (Tmax) in degree Celsius.
- River stage (Stage_m) in meter.
- Potential evaporation (ET) computed with Hargreaves method in mm/d.

## Calibration and testing data

The head data are split into a training and testing period. Data are provided for the training/calibration period. Participants have to provide a simulated time 
series for the entire period: 

- **Training period:** 2002-03-01 to 2016-12-31
- **Testing period:** 2017-01-01 to 2022-05-31

## GitHub Discussion about this data

If you have any questions or comments about the data provided for this location, please post in the [GitHub 
Discussion related to this well](https://github.com/gwmodeling/challenge/discussions/4). This way, all participants 
will have the same information.

## Data Acknowledgments

- We acknowledge the United States Geological Survey (www.usgs.gov) for providing the river stage and hydraulic head time series.
- We acknowledge the National Centers for Environmental Information of the National Oceanic and Atmospheric Administration (www.noaa.gov) for providing the precipitation, minimum temperature, and maximum temperature time series.