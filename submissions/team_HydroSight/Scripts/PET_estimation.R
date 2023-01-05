# This script is used to estimate PET with different methods
# by using the R package "Evapotranspiration" .
#
# Reference: Guo Danlu, et al. 2016, Environmental Modelling & Software
# An R package for modelling actual, potential and reference evapotranspiration.
#
#
# Author: Xinyang Fan
# Date: 2023.01.05
# R 3.6.3


setwd("D:\\GW_modelling_challenge\\Climate\\PET_variants")

# library the package
library(Evapotranspiration)
library(lubridate)
library(zoo)


# PET calculation for Germany###################################################

# load data 
climate <- read.csv('D:\\GW_modelling_challenge\\Backup\\Raw_data\\climate_Germany_allvariables_github.csv')

# write input data required by the ET package
Date.daily <- ISOdate(climate$year, climate$month, climate$day)
Date.daily <- as.Date(Date.daily, "%d.%m.%Y", tz = "UTC")

JulianDay <- as.POSIXlt(Date.daily, format = "%y.%m.%d")
JulianDay <- JulianDay$yday

yearmonth <- cbind(climate$year, climate$month)
yearmonth <- unique(yearmonth)
imonth <- yearmonth[,2]

yearmonth2 <- cbind(yearmonth, replicate(length(yearmonth[,1]), 1))
yearmonth2 <- ISOdate(yearmonth2[,1], yearmonth2[,2], yearmonth2[,3])
ndays <- days_in_month(yearmonth2)

Date.monthly <- as.Date(yearmonth2, "%d.%m.%Y", tz="UTC")

Tmax <- climate$tx
Tmin <- climate$tn

RHmax <- climate$hu    # %
RHmin <- climate$hu

Rs <- climate$qq    # W/m2
Rs <- Rs * 0.0864     # 1 W/m2 = 0.0864 MJ/m2/day

uz <- climate$fg   # m/s


# write the data list
datalist <- list(
  Date.daily = Date.daily,
  Date.monthly = Date.monthly,
  J = as.zoo(JulianDay),
  i = as.zoo(imonth),
  ndays = as.zoo(ndays),
  Tmax = as.zoo(Tmax),
  Tmin = as.zoo(Tmin),
  RHmax = as.zoo(RHmax),
  RHmin = as.zoo(RHmin),
  uz = uz,
  Rs = Rs
)

# load data constants
data("constants")

# calculate Penman PET: T, RH, Rs, uz
PET_Penman_short <- ET.PenmanMonteith(datalist, constants, ts="daily", solar = "data", wind = "yes", crop = "short", 
                         message="yes", AdditionalStats="yes", save.csv="no")

# calculate Penman PET: T, RH, Rs, uz
PET_Penman_tall <- ET.PenmanMonteith(datalist, constants, ts="daily", solar = "data", wind = "yes", crop = "tall", 
                                message="yes", AdditionalStats="yes", save.csv="no")

# calculate Turc PET; T, Rs
PET_Turc <- ET.Turc(datalist, constants, ts = "daily", solar = "data", humid = T, message = "yes", 
                    AdditionalStats = "yes", save.csv = "no")

# calculate Jensen-Haise PET: T, Rs
PET_Jensen <- ET.JensenHaise(datalist, constants, ts="daily", solar = "data",
                             message="yes", AdditionalStats="yes", save.csv="no")

# calculate Makkink PET: T, Rs
PET_Makkink <- ET.Makkink(datalist, constants, ts="daily", solar = "data",
                          message="yes", AdditionalStats="yes", save.csv="no")

# calculate Hargreaves Samani PET: T
PET_Hargreaves <- ET.HargreavesSamani(datalist, constants, ts = "daily", message = "yes", 
                    AdditionalStats = "yes", save.csv = "no")

# calculate Priestley-Taylor PET: T, RH, Rs
PET_Priestley <- ET.PriestleyTaylor(datalist, constants, ts="daily", solar = "data", alpha = 0.23,
                                    message="yes", AdditionalStats="yes", save.csv="no")

ETComparison(PET_Penman_short, PET_Penman_tall, results3 = PET_Turc, results4 = PET_Jensen, results5 = PET_Makkink, results6 = PET_Hargreaves, 
             results7 = PET_Priestley, 
             labs = c("PenmanShort", "PenmanTall", "Turc", "Jensen", "Makkink", "Hargreaves", "Priestley"),
             type = "Monthly", ylim = c(0,400))


# Summarise the PET in one table
PET_daily <- data.frame(climate$year, climate$month, climate$day, PET_Penman_short$ET.Daily, PET_Penman_tall$ET.Daily, PET_Turc$ET.Daily, PET_Jensen$ET.Daily, PET_Makkink$ET.Daily,
                        PET_Hargreaves$ET.Daily, PET_Priestley$ET.Daily)
colnames(PET_daily) <- c("year","month","day","PET_Penman_short", "PET_Penman_tall", "PET_Turc", "PET_Jensen", "PET_Makkink",
                         "PET_Hargreaves", "PET_Priestley")

# write.csv(PET_daily, file = "PET_daily_Germany.csv", row.names = F)




# PET calculation for Netherland ###############################################

# load data 
climate <- read.csv('D:\\GW_modelling_challenge\\Backup\\Raw_data\\climate_Netherland_allvariables_github.csv')

# write input data required by the ET package
Date.daily <- ISOdate(climate$year, climate$month, climate$day)
Date.daily <- as.Date(Date.daily, "%d.%m.%Y", tz = "UTC")

JulianDay <- as.POSIXlt(Date.daily, format = "%y.%m.%d")
JulianDay <- JulianDay$yday

yearmonth <- cbind(climate$year, climate$month)
yearmonth <- unique(yearmonth)
imonth <- yearmonth[,2]

yearmonth2 <- cbind(yearmonth, replicate(length(yearmonth[,1]), 1))
yearmonth2 <- ISOdate(yearmonth2[,1], yearmonth2[,2], yearmonth2[,3])
ndays <- days_in_month(yearmonth2)

Date.monthly <- as.Date(yearmonth2, "%d.%m.%Y", tz="UTC")

Tmax <- climate$tx
Tmin <- climate$tn

RHmax <- climate$hu    # %
RHmin <- climate$hu

Rs <- climate$qq    # W/m2
Rs <- Rs * 0.0864     # 1 W/m2 = 0.0864 MJ/m2/day

uz <- climate$fg   # m/s


# write the data list
datalist <- list(
  Date.daily = Date.daily,
  Date.monthly = Date.monthly,
  J = as.zoo(JulianDay),
  i = as.zoo(imonth),
  ndays = as.zoo(ndays),
  Tmax = as.zoo(Tmax),
  Tmin = as.zoo(Tmin),
  RHmax = as.zoo(RHmax),
  RHmin = as.zoo(RHmin),
  uz = uz,
  Rs = Rs
)

# load data constants
data("constants")

# calculate Penman PET: T, RH, Rs, uz
PET_Penman_short <- ET.PenmanMonteith(datalist, constants, ts="daily", solar = "data", wind = "yes", crop = "short", 
                                      message="yes", AdditionalStats="yes", save.csv="no")

# calculate Penman PET: T, RH, Rs, uz
PET_Penman_tall <- ET.PenmanMonteith(datalist, constants, ts="daily", solar = "data", wind = "yes", crop = "tall", 
                                     message="yes", AdditionalStats="yes", save.csv="no")

# calculate Turc PET; T, Rs
PET_Turc <- ET.Turc(datalist, constants, ts = "daily", solar = "data", humid = T, message = "yes", 
                    AdditionalStats = "yes", save.csv = "no")

# calculate Jensen-Haise PET: T, Rs
PET_Jensen <- ET.JensenHaise(datalist, constants, ts="daily", solar = "data",
                             message="yes", AdditionalStats="yes", save.csv="no")

# calculate Makkink PET: T, Rs
PET_Makkink <- ET.Makkink(datalist, constants, ts="daily", solar = "data",
                          message="yes", AdditionalStats="yes", save.csv="no")

# calculate Hargreaves Samani PET: T
PET_Hargreaves <- ET.HargreavesSamani(datalist, constants, ts = "daily", message = "yes", 
                                      AdditionalStats = "yes", save.csv = "no")

# calculate Priestley-Taylor PET: T, RH, Rs
PET_Priestley <- ET.PriestleyTaylor(datalist, constants, ts="daily", solar = "data", alpha = 0.23,
                                    message="yes", AdditionalStats="yes", save.csv="no")

ETComparison(PET_Penman_short, PET_Penman_tall, results3 = PET_Turc, results4 = PET_Jensen, results5 = PET_Makkink, results6 = PET_Hargreaves, 
             results7 = PET_Priestley, 
             labs = c("PenmanShort", "PenmanTall", "Turc", "Jensen", "Makkink", "Hargreaves", "Priestley"),
             type = "Monthly", ylim = c(0,400))


# Summarise the PET in one table

PET_daily <- data.frame(climate$year, climate$month, climate$day, PET_Penman_short$ET.Daily, PET_Penman_tall$ET.Daily, PET_Turc$ET.Daily, PET_Jensen$ET.Daily, PET_Makkink$ET.Daily,
                        PET_Hargreaves$ET.Daily, PET_Priestley$ET.Daily)
colnames(PET_daily) <- c("year","month","day", "PET_Penman_short", "PET_Penman_tall", "PET_Turc", "PET_Jensen", "PET_Makkink",
                         "PET_Hargreaves", "PET_Priestley")

# write.csv(PET_daily, file = "PET_daily_Netherland.csv", row.names = F)




# PET calculation for Sweden1 ##################################################

# load data, note, Rs: first 15 daily raw data are missing
climate <- read.csv('D:\\GW_modelling_challenge\\Backup\\Raw_data\\climate_Sweden_1_allvariables_github.csv')


# write input data required by the ET package
Date.daily <- ISOdate(climate$year, climate$month, climate$day)
Date.daily <- as.Date(Date.daily, "%d.%m.%Y", tz = "UTC")

JulianDay <- as.POSIXlt(Date.daily, format = "%y.%m.%d")
JulianDay <- JulianDay$yday

yearmonth <- cbind(climate$year, climate$month)
yearmonth <- unique(yearmonth)
imonth <- yearmonth[,2]

yearmonth2 <- cbind(yearmonth, replicate(length(yearmonth[,1]), 1))
yearmonth2 <- ISOdate(yearmonth2[,1], yearmonth2[,2], yearmonth2[,3])
ndays <- days_in_month(yearmonth2)

Date.monthly <- as.Date(yearmonth2, "%d.%m.%Y", tz="UTC")

Tmax <- climate$tx
Tmin <- climate$tn

RHmax <- climate$hu    # %
RHmin <- climate$hu

Rs <- climate$qq    # W/m2
Rs <- Rs * 0.0864     # 1 W/m2 = 0.0864 MJ/m2/day

uz <- climate$fg   # m/s


# write the data list
datalist <- list(
  Date.daily = Date.daily,
  Date.monthly = Date.monthly,
  J = as.zoo(JulianDay),
  i = as.zoo(imonth),
  ndays = as.zoo(ndays),
  Tmax = as.zoo(Tmax),
  Tmin = as.zoo(Tmin),
  RHmax = as.zoo(RHmax),
  RHmin = as.zoo(RHmin),
  uz = uz,
  Rs = Rs
)

# load data constants
data("constants")

# calculate Penman PET: T, RH, Rs, uz
PET_Penman_short <- ET.PenmanMonteith(datalist, constants, ts="daily", solar = "data", wind = "yes", crop = "short", 
                                      message="yes", AdditionalStats="yes", save.csv="no")

# calculate Penman PET: T, RH, Rs, uz
PET_Penman_tall <- ET.PenmanMonteith(datalist, constants, ts="daily", solar = "data", wind = "yes", crop = "tall", 
                                     message="yes", AdditionalStats="yes", save.csv="no")

# calculate Turc PET; T, Rs
PET_Turc <- ET.Turc(datalist, constants, ts = "daily", solar = "data", humid = T, message = "yes", 
                    AdditionalStats = "yes", save.csv = "no")

# calculate Jensen-Haise PET: T, Rs
PET_Jensen <- ET.JensenHaise(datalist, constants, ts="daily", solar = "data",
                             message="yes", AdditionalStats="yes", save.csv="no")

# calculate Makkink PET: T, Rs
PET_Makkink <- ET.Makkink(datalist, constants, ts="daily", solar = "data",
                          message="yes", AdditionalStats="yes", save.csv="no")

# calculate Hargreaves Samani PET: T
PET_Hargreaves <- ET.HargreavesSamani(datalist, constants, ts = "daily", message = "yes", 
                                      AdditionalStats = "yes", save.csv = "no")

# calculate Priestley-Taylor PET: T, RH, Rs
PET_Priestley <- ET.PriestleyTaylor(datalist, constants, ts="daily", solar = "data", alpha = 0.23,
                                    message="yes", AdditionalStats="yes", save.csv="no")


# Summarise the PET in one table
PET_daily <- data.frame(climate$year, climate$month, climate$day, PET_Penman_short$ET.Daily, PET_Penman_tall$ET.Daily, PET_Turc$ET.Daily, PET_Jensen$ET.Daily, PET_Makkink$ET.Daily,
                        PET_Hargreaves$ET.Daily, PET_Priestley$ET.Daily)
colnames(PET_daily) <- c("year","month","day","PET_Penman_short", "PET_Penman_tall", "PET_Turc", "PET_Jensen", "PET_Makkink",
                         "PET_Hargreaves", "PET_Priestley")

# write.csv(PET_daily, file = "PET_daily_Sweden2.csv", row.names = F)




# PET calculation for Sweden2 ##################################################

# load data, note, Rs: first 19 daily raw data are missing
climate <- read.csv('D:\\GW_modelling_challenge\\Backup\\Raw_data\\climate_Sweden_2_allvariables_github.csv')

# write input data required by the ET package
Date.daily <- ISOdate(climate$year, climate$month, climate$day)
Date.daily <- as.Date(Date.daily, "%d.%m.%Y", tz = "UTC")

JulianDay <- as.POSIXlt(Date.daily, format = "%y.%m.%d")
JulianDay <- JulianDay$yday

yearmonth <- cbind(climate$year, climate$month)
yearmonth <- unique(yearmonth)
imonth <- yearmonth[,2]

yearmonth2 <- cbind(yearmonth, replicate(length(yearmonth[,1]), 1))
yearmonth2 <- ISOdate(yearmonth2[,1], yearmonth2[,2], yearmonth2[,3])
ndays <- days_in_month(yearmonth2)

Date.monthly <- as.Date(yearmonth2, "%d.%m.%Y", tz="UTC")

Tmax <- climate$tx
Tmin <- climate$tn

RHmax <- climate$hu    # %
RHmin <- climate$hu

Rs <- climate$qq    # W/m2
Rs <- Rs * 0.0864     # 1 W/m2 = 0.0864 MJ/m2/day

uz <- climate$fg   # m/s


# write the data list
datalist <- list(
  Date.daily = Date.daily,
  Date.monthly = Date.monthly,
  J = as.zoo(JulianDay),
  i = as.zoo(imonth),
  ndays = as.zoo(ndays),
  Tmax = as.zoo(Tmax),
  Tmin = as.zoo(Tmin),
  RHmax = as.zoo(RHmax),
  RHmin = as.zoo(RHmin),
  uz = uz,
  Rs = Rs
)

# load data constants
data("constants")

# calculate Penman PET: T, RH, Rs, uz
PET_Penman_short <- ET.PenmanMonteith(datalist, constants, ts="daily", solar = "data", wind = "yes", crop = "short", 
                                      message="yes", AdditionalStats="yes", save.csv="no")

# calculate Penman PET: T, RH, Rs, uz
PET_Penman_tall <- ET.PenmanMonteith(datalist, constants, ts="daily", solar = "data", wind = "yes", crop = "tall", 
                                     message="yes", AdditionalStats="yes", save.csv="no")

# calculate Turc PET; T, Rs
PET_Turc <- ET.Turc(datalist, constants, ts = "daily", solar = "data", humid = T, message = "yes", 
                    AdditionalStats = "yes", save.csv = "no")

# calculate Jensen-Haise PET: T, Rs
PET_Jensen <- ET.JensenHaise(datalist, constants, ts="daily", solar = "data",
                             message="yes", AdditionalStats="yes", save.csv="no")

# calculate Makkink PET: T, Rs
PET_Makkink <- ET.Makkink(datalist, constants, ts="daily", solar = "data",
                          message="yes", AdditionalStats="yes", save.csv="no")

# calculate Hargreaves Samani PET: T
PET_Hargreaves <- ET.HargreavesSamani(datalist, constants, ts = "daily", message = "yes", 
                                      AdditionalStats = "yes", save.csv = "no")

# calculate Priestley-Taylor PET: T, RH, Rs
PET_Priestley <- ET.PriestleyTaylor(datalist, constants, ts="daily", solar = "data", alpha = 0.23,
                                    message="yes", AdditionalStats="yes", save.csv="no")

# Summarise the PET in one table
PET_daily <- data.frame(climate$year, climate$month, climate$day, PET_Penman_short$ET.Daily, PET_Penman_tall$ET.Daily, PET_Turc$ET.Daily, PET_Jensen$ET.Daily, PET_Makkink$ET.Daily,
                        PET_Hargreaves$ET.Daily, PET_Priestley$ET.Daily)
colnames(PET_daily) <- c("year","month","day", "PET_Penman_short", "PET_Penman_tall", "PET_Turc", "PET_Jensen", "PET_Makkink",
                         "PET_Hargreaves", "PET_Priestley")

# write.csv(PET_daily, file = "PET_daily_Sweden2.csv", row.names = F)




# PET calculation for USA ######################################################

# load data, only Hargreaves can be calculated as only T available.
climate <- read.csv('D:\\GW_modelling_challenge\\Backup\\Raw_data\\climate_USA_allvariables_github.csv')

# write input data required by the ET package
Date.daily <- ISOdate(climate$year, climate$month, climate$day)
Date.daily <- as.Date(Date.daily, "%d.%m.%Y", tz = "UTC")

JulianDay <- as.POSIXlt(Date.daily, format = "%y.%m.%d")
JulianDay <- JulianDay$yday

yearmonth <- cbind(climate$year, climate$month)
yearmonth <- unique(yearmonth)
imonth <- yearmonth[,2]

yearmonth2 <- cbind(yearmonth, replicate(length(yearmonth[,1]), 1))
yearmonth2 <- ISOdate(yearmonth2[,1], yearmonth2[,2], yearmonth2[,3])
ndays <- days_in_month(yearmonth2)

Date.monthly <- as.Date(yearmonth2, "%d.%m.%Y", tz="UTC")

Tmax <- climate$TMAX
Tmin <- climate$TMIN

# write the data list
datalist <- list(
  Date.daily = Date.daily,
  Date.monthly = Date.monthly,
  J = as.zoo(JulianDay),
  i = as.zoo(imonth),
  ndays = as.zoo(ndays),
  Tmax = as.zoo(Tmax),
  Tmin = as.zoo(Tmin),
  RHmax = NULL,
  RHmin = NULL,
  uz = NULL,
  Rs = NULL
)

# load data constants
data("constants")

# calculate Hargreaves Samani PET: T
PET_Hargreaves <- ET.HargreavesSamani(datalist, constants, ts = "daily", message = "yes", 
                                      AdditionalStats = "yes", save.csv = "no")

# write the PET file
PET_daily <- data.frame(climate$year, climate$month, climate$day, PET_Hargreaves$ET.Daily)
colnames(PET_daily) <- c("year","month","day","PET_Hargreaves")

# write.csv(PET_daily, file = "PET_daily_USA.csv", row.names = F)











