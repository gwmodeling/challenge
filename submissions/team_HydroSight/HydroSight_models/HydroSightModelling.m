% This script builds and runs HydroSight models for the Groundwater
% Challenge. The modelling was done using HydroSIght version 1.42.3
% Importantly, the modelling was done using the HydroSight GUI. 
% This script uses the identical model settings as developed within
% the GUI and is provided to meet the submission requirements.
% 
% To run this script, download the HydroSight source code for version 1.42.3
% from: https://github.com/peterson-tim-j/HydroSight.

% Import head data files
head_Germany = readtable("head_karst_confined_Germany.csv");
head_Netherlands = readtable("head_sand_unconfined_Netherland.csv");
head_Sweden1 = readtable("head_fractured_unconfined_Sweden_1.csv");
head_Sweden2 = readtable("head_till_unconfined_Sweden_2.csv");
head_USA = readtable("head_bedrock_confined_USA.csv");

% Import forcing data files
forcing_Germany = readtable("climate_PET_Makkink_Snowmelt_Germany_29.csv");
forcing_Netherlands = readtable("climate_PET_Makkink_Snowmelt_Netherland_5.csv");
forcing_Sweden1 = readtable("climate_Sweden_1_rawP_PET_Makkink_Tmean.csv");
forcing_Sweden2 = readtable("climate_PET_Penman_Snowmelt_Sweden2_24.csv");
forcing_USA = readtable("climate_USA_rawP_PET_Hargreaves_Tmean.csv");

% Import coordinates files - note they contain dummy data
coords_Germany = table2cell(readtable("coordinates_Germany.csv"));
coords_Netherlands = table2cell(readtable("coordinates_Netherland.csv"));
coords_Sweden1 = table2cell(readtable("coordinates_Sweden_1.csv"));
coords_Sweden2 = table2cell(readtable("coordinates_Sweden_2.csv"));
coords_USA = table2cell(readtable("coordinates_USA.csv"));

% Define the model structures. Note, are copied from the GUI and are best
% derived using it.
modelOptions_Germany = { 'recharge'  'weightingfunction'   'responseFunction_Pearsons'; 'recharge'  'forcingdata'  { 'transformfunction'   'climateTransform_soilMoistureModels_2layer_v2'; 'forcingdata'  { 'precip'  'precip_mm' ; 'et'  'ET_mm' ; 'TreeFraction'  '(none)' ;}; 'options'  { 'SMSC'  2.000000  'Calib.' ; 'SMSC_trees'  2.000000  'Fixed' ; 'treeArea_frac'  0.500000  'Fixed' ; 'S_initialfrac'  0.500000  'Fixed' ; 'k_infilt'  Inf  'Fixed' ; 'k_sat'  1.000000  'Calib.' ; 'bypass_frac'  0.000000  'Fixed' ; 'alpha'  0.000000  'Fixed' ; 'beta'  0.500000  'Calib.' ; 'gamma'  0.000000  'Calib.' ; 'SMSC_deep'  2.000000  'Calib.' ; 'SMSC_deep_trees'  2.000000  'Fixed' ; 'S_initialfrac_deep'  0.500000  'Fixed' ; 'k_sat_deep'  1.000000  'Calib.' ; 'beta_deep'  0.500000  'Calib.' ;}; 'outputdata'   'drainage_deep' }; 'gwet'  'weightingfunction'   'responseFunction_PearsonsNegative'; 'gwet'  'forcingdata'  { 'transformfunction'  'climateTransform_soilMoistureModels_2layer_v2' ; 'outputdata'  'evap_gw_potential' ;};};
modelOptions_Netherlands = { 'recharge'  'weightingfunction'   'responseFunction_Pearsons'; 'recharge'  'forcingdata'  { 'transformfunction'   'climateTransform_soilMoistureModels_2layer_v2'; 'forcingdata'  { 'precip'  'precip_mm' ; 'et'  'ET_mm' ; 'TreeFraction'  '(none)' ;}; 'options'  { 'SMSC'  2.000000  'Calib.' ; 'SMSC_trees'  2.000000  'Fixed' ; 'treeArea_frac'  0.500000  'Fixed' ; 'S_initialfrac'  0.500000  'Fixed' ; 'k_infilt'  Inf  'Fixed' ; 'k_sat'  1.000000  'Calib.' ; 'bypass_frac'  0.000000  'Fixed' ; 'alpha'  0.000000  'Fixed' ; 'beta'  0.500000  'Calib.' ; 'gamma'  0.000000  'Calib.' ; 'SMSC_deep'  2.000000  'Calib.' ; 'SMSC_deep_trees'  2.000000  'Fixed' ; 'S_initialfrac_deep'  0.500000  'Fixed' ; 'k_sat_deep'  1.000000  'Calib.' ; 'beta_deep'  0.500000  'Calib.' ;}; 'outputdata'   'drainage_deep' }; 'gwet'  'weightingfunction'   'responseFunction_PearsonsNegative'; 'gwet'  'forcingdata'  { 'transformfunction'  'climateTransform_soilMoistureModels_2layer_v2' ; 'outputdata'  'evap_gw_potential' ;};};
modelOptions_Sweden1 = { 'recharge', 'weightingfunction',  'responseFunction_Pearsons'; 'recharge', 'forcingdata', { 'transformfunction',  'climateTransform_soilMoistureModels'; 'forcingdata', { 'precip', 'precip_mm' ; 'et', 'ET_mm' ; 'TreeFraction', '(none)' ; 'temperature', 'Tmean' ;}; 'options', { 'SMSC', 2.000000, 'Calib.' ; 'SMSC_trees', 2.000000, 'Fixed' ; 'treeArea_frac', 0.500000, 'Fixed' ; 'S_initialfrac', 1.000000, 'Fixed' ; 'k_infilt', Inf, 'Fixed' ; 'k_sat', 1.000000, 'Calib.' ; 'bypass_frac', 0.000000, 'Fixed' ; 'interflow_frac', 0.000000, 'Fixed' ; 'alpha', 1.000000, 'Fixed' ; 'beta', 0.500000, 'Calib.' ; 'gamma', 0.000000, 'Calib.' ; 'eps', 0.000000, 'Fixed' ; 'DDF', 5.000000, 'Calib.' ; 'melt_threshold', 3.000000, 'Calib.' ;}; 'outputdata' , 'drainage' }; 'gwet', 'weightingfunction',  'responseFunction_PearsonsNegative'; 'gwet', 'forcingdata', { 'transformfunction', 'climateTransform_soilMoistureModels' ; 'outputdata', 'evap_gw_potential' ;};}; 
modelOptions_Sweden2 = { 'recharge'  'weightingfunction'   'responseFunction_Pearsons'; 'recharge'  'forcingdata'  { 'transformfunction'   'climateTransform_soilMoistureModels_2layer_v2'; 'forcingdata'  { 'precip'  'precip_mm' ; 'et'  'ET_mm' ; 'TreeFraction'  '(none)' ;}; 'options'  { 'SMSC'  2.000000  'Calib.' ; 'SMSC_trees'  2.000000  'Fixed' ; 'treeArea_frac'  0.500000  'Fixed' ; 'S_initialfrac'  0.500000  'Fixed' ; 'k_infilt'  Inf  'Fixed' ; 'k_sat'  1.000000  'Calib.' ; 'bypass_frac'  0.000000  'Fixed' ; 'alpha'  0.000000  'Fixed' ; 'beta'  0.500000  'Calib.' ; 'gamma'  0.000000  'Calib.' ; 'SMSC_deep'  2.000000  'Calib.' ; 'SMSC_deep_trees'  2.000000  'Fixed' ; 'S_initialfrac_deep'  0.500000  'Fixed' ; 'k_sat_deep'  1.000000  'Calib.' ; 'beta_deep'  0.500000  'Calib.' ;}; 'outputdata'   'drainage_deep' }; 'gwet'  'weightingfunction'   'responseFunction_PearsonsNegative'; 'gwet'  'forcingdata'  { 'transformfunction'  'climateTransform_soilMoistureModels_2layer_v2' ; 'outputdata'  'evap_gw_potential' ;};};
modelOptions_USA = { 'recharge', 'weightingfunction',  'responseFunction_Pearsons'; 'recharge', 'forcingdata', { 'transformfunction',  'climateTransform_soilMoistureModels'; 'forcingdata', { 'precip', 'precip_mm' ; 'et', 'ET_mm' ; 'TreeFraction', '(none)' ; 'temperature', 'Tmean' ;}; 'options', { 'SMSC', 2.000000, 'Calib.' ; 'SMSC_trees', 2.000000, 'Fixed' ; 'treeArea_frac', 0.500000, 'Fixed' ; 'S_initialfrac', 1.000000, 'Fixed' ; 'k_infilt', Inf, 'Fixed' ; 'k_sat', 1.000000, 'Calib.' ; 'bypass_frac', 0.000000, 'Fixed' ; 'interflow_frac', 0.000000, 'Fixed' ; 'alpha', 1.000000, 'Fixed' ; 'beta', 0.500000, 'Calib.' ; 'gamma', 0.000000, 'Calib.' ; 'eps', 0.000000, 'Fixed' ; 'DDF', 5.000000, 'Calib.' ; 'melt_threshold', 3.000000, 'Calib.' ;}; 'outputdata' , 'drainage' }; 'gwet', 'weightingfunction',  'responseFunction_PearsonsNegative'; 'gwet', 'forcingdata', { 'transformfunction', 'climateTransform_soilMoistureModels' ; 'outputdata', 'evap_gw_potential' ;};};

minHeadTimestep = 1; %days
modelType = 'model_TFN';

% Build HydroSight models
model_Germany = HydroSightModel('Germany', 'Germany', modelType , head_Germany{:,2:end}, minHeadTimestep, forcing_Germany, coords_Germany, modelOptions_Germany);
model_Netherland = HydroSightModel('Netherland', 'Netherland', modelType , head_Netherlands{:,2:end}, minHeadTimestep, forcing_Netherlands, coords_Netherlands, modelOptions_Netherlands);
model_Sweden1 = HydroSightModel('Sweden1', 'Sweden1', modelType , head_Sweden1{:,2:end}, minHeadTimestep, forcing_Sweden1, coords_Sweden1, modelOptions_Sweden1);
model_Sweden2 = HydroSightModel('Sweden2', 'Sweden2', modelType , head_Sweden2{:,2:end}, minHeadTimestep, forcing_Sweden2, coords_Sweden2, modelOptions_Sweden2);
model_USA = HydroSightModel('USA', 'USA', modelType , head_USA{:,2:end}, minHeadTimestep, forcing_USA, coords_USA, modelOptions_USA);

% Define Calibration settings
calibMethod = 'CMAES';
calibMethodSetting.MaxFunEvals = inf;
calibMethodSetting.PopSize= 4;
calibMethodSetting.TolFun= 1E-8;
calibMethodSetting.TolX= 1E-7;
calibMethodSetting.Restarts= 2;
calibMethodSetting.Sigma= 0.33;
calibMethodSetting.Seed= 913375;

% Define calibration period start and end dates
calib_start_end_Germany = [model_Germany.model.inputData.head(1,1), model_Germany.model.inputData.head(end,1)];
calib_start_end_Netherland = [model_Netherland.model.inputData.head(1,1), model_Netherland.model.inputData.head(end,1)];
calib_start_end_Sweden1 = [model_Sweden1.model.inputData.head(1,1), model_Sweden1.model.inputData.head(end,1)];
calib_start_end_Sweden2 = [model_Sweden2.model.inputData.head(1,1), model_Sweden2.model.inputData.head(end,1)];
calib_start_end_USA = [model_USA.model.inputData.head(1,1), model_USA.model.inputData.head(end,1)];

% Calibrate Models
calibrateModel( model_Germany, [], calib_start_end_Germany(1), calib_start_end_Germany(2), calibMethod,  calibMethodSetting);
calibrateModel( model_Netherland, [], calib_start_end_Netherland(1), calib_start_end_Netherland(2), calibMethod,  calibMethodSetting);
calibrateModel( model_Sweden1, [], calib_start_end_Sweden1(1), calib_start_end_Sweden1(2), calibMethod,  calibMethodSetting);
calibrateModel( model_Sweden2, [], calib_start_end_Sweden2(1), calib_start_end_Sweden2(2), calibMethod,  calibMethodSetting);
calibrateModel( model_USA, [], calib_start_end_USA(1), calib_start_end_USA(2), calibMethod,  calibMethodSetting);

% Plot Calibration results
calibrateModelPlotResults(model_Germany,[]);
calibrateModelPlotResults(model_Netherland,[]);
calibrateModelPlotResults(model_Sweden1,[]);
calibrateModelPlotResults(model_Sweden2,[]);
calibrateModelPlotResults(model_USA,[]);

% Get the forcing data time points for the simulation
time_point_Germany = model_Germany.model.inputData.forcingData(:,1);
time_point_Netherlands = model_Netherland.model.inputData.forcingData(:,1);
time_point_Sweden1 = model_Sweden1.model.inputData.forcingData(:,1);
time_point_Sweden2 = model_Sweden2.model.inputData.forcingData(:,1);
time_point_USA = model_USA.model.inputData.forcingData(:,1);

% Filter simulation time points to start at the first head obs.
time_point_Germany = time_point_Germany(time_point_Germany>=calib_start_end_Germany(1)); 
time_point_Netherlands = time_point_Netherlands(time_point_Netherlands>=calib_start_end_Netherland(1)); 
time_point_Sweden1 = time_point_Sweden1(time_point_Sweden1>=calib_start_end_Sweden1(1)); 
time_point_Sweden2 = time_point_Sweden2(time_point_Sweden2>=calib_start_end_Sweden2(1)); 
time_point_USA = time_point_USA(time_point_USA>=calib_start_end_USA(1)); 

% Define settings for simulations
newForcingData = [];
simulationLabel = 'default simulation';
doKrigingOnResiduals = false;

% Simulate daily head over the calibration and prediction periods
solveModel(model_Germany, time_point_Germany, newForcingData, simulationLabel, doKrigingOnResiduals);
solveModel(model_Netherland, time_point_Netherlands, newForcingData, simulationLabel, doKrigingOnResiduals);
solveModel(model_Sweden1, time_point_Sweden1, newForcingData, simulationLabel, doKrigingOnResiduals);
solveModel(model_Sweden2, time_point_Sweden2, newForcingData, simulationLabel, doKrigingOnResiduals);
solveModel(model_USA, time_point_USA, newForcingData, simulationLabel, doKrigingOnResiduals);

% Plot simulation results
solveModelPlotResults(model_Germany, simulationLabel, []);
solveModelPlotResults(model_Netherland, simulationLabel, []);
solveModelPlotResults(model_Sweden1, simulationLabel, []);
solveModelPlotResults(model_Sweden2, simulationLabel, []);
solveModelPlotResults(model_USA, simulationLabel, []);


