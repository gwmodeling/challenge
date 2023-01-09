% This script is used to calculate the time-series liquid water by using
% the observed precipitation (rain and snow) and Tmean.
%
% Method: Degree-Day-Factor method
%
% Reference: Calli et al. 2022, JoH.
% Contribution of the satellite-data driven snow routine to a karst
% hydrological model.
%
% Author: Xinyang Fan
% Date: 2022.12.04
% MATLAB R2022a

clc; clear all;

%% load daily observed climate data

climate = readtable('D:\GW_modelling_challenge\To_submit_GitHub\Raw_climate\climate_Germany_allvariables_github.csv');

% get the Precip
P = climate.rr;
% get the Tmean
Tmean = climate.tg;


%% set the range of the Parameters of the degree-day method

% get the number of days of the climate forcing
ndays = length(P);

% degree day factor (DDF) to calibrate
DDF_min = 0.1;
DDF_max = 3;
stepsize = 0.3;
DDF = (DDF_min:stepsize:DDF_max)';

% threshold temperature (Tm) at which snowmelt begins
Tm_min = -1;
Tm_max = 1;
stepsize = 1;
Tm = (Tm_min:stepsize:Tm_max)';


%% Degree-Day method to calculate time-series liquid water (Lw)

nruns = length(DDF) * length(Tm);

% claim empty matrix
Melt = nan(ndays,nruns);   % melted amount of snow
Snow = nan(ndays,nruns);   % snow accumulation
Lw = nan(ndays,nruns);     % liquid water: time-series transformed precipitation that infiltrates into the soil

% set the first day as 0
Lw(1,:) = 0;
Melt(1,:) = 0;
Snow(1,:) = 0;

for m = 1:length(DDF)
    for n = 1:length(Tm)
      j = length(Tm) * (m-1) + n;
        for i = 1:(ndays-1)
            if Tmean(i+1,1) <= Tm(n,1)
                % snow melt 
                Melt(i+1,j) = 0;
                % snow
                Snow(i+1,j) = Snow(i,j) + P(i+1,1);
                % liquid water
                Lw(i+1,j) = 0;
            else
                % snow melt
                Melt(i+1,j) = DDF(m,1) * (Tmean(i+1,1) - Tm(n,1));
                % snow
                Snow(i+1,j) = max(Snow(i,j) - Melt(i+1,j),0);
                % liquid water
                Lw(i+1,j) = P(i+1,1) + min(Snow(i,j),Melt(i+1,j));
            end
        end
    end
end


%% write the parameter table of DDF and Tm

parameter_matrix = nan(nruns, 3);
% first column represents the number of runs (corresponding to the column number of the Lw array)
parameter_matrix(:,1) = (1:1:nruns)';

for m = 1:length(DDF)
    for n = 1:length(Tm)
      j = length(Tm) * (m-1) + n;

      % second column contains the DDF factor
      parameter_matrix(j,2) = DDF(m,1);
      % third column contains the Tm
      parameter_matrix(j,3) = Tm(n,1);
    end
end

parameter_matrix = array2table(parameter_matrix, "VariableNames", ["runs", "DDF", "Tm"]);

% save the parameter table
% writetable(parameter_matrix, 'D:\GW_modelling_challenge\Climate\Snowmelt\Parameter_table.csv')


%% save the Lw, Melt, and Snow

% add the dates
Lw = [climate.year climate.month climate.day Lw];
Snow = [climate.year climate.month climate.day Snow];
Melt = [climate.year climate.month climate.day Melt];

% write the table
Lw = array2table(Lw);
Snow = array2table(Snow);
Melt = array2table(Melt);

% writetable(Lw, 'D:\GW_modelling_challenge\Climate\Snowmelt\Lw_DegreeDayCal_daily_Netherlands.csv');
% writetable(Snow, 'D:\GW_modelling_challenge\Climate\Snowmelt\SnowWaterEqui_DegreeDayCal_daily.csv');
% writetable(Melt, 'D:\GW_modelling_challenge\Climate\Snowmelt\Melt_DegreeDayCal_daily.csv');









