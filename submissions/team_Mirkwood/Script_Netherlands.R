library(tidyverse)
library(tidymodels)
library(lubridate)
library(vip)
library(zoo)

#####Read predictor data and calculate movimg average/sum (right aligned) for all variables
#with widths of 7, 30, 100, 365 days 

df_in <-  read_csv('data/Netherlands/input_data.csv') %>% 
  rename(Date = time) %>% 
  mutate(rr_agg100 = rollsumr(rr, 100, fill = NA)) %>%
  mutate(rr_agg365 = rollsumr(rr, 365, fill = NA)) %>%
  mutate(rr_agg7 = rollsumr(rr, 7, fill = NA)) %>% 
  mutate(rr_agg30 = rollsumr(rr, 30, fill = NA)) %>% 
  mutate(et_agg30 = rollsumr(rr, 30, fill = NA)) %>%
  mutate(et_agg7 = rollsumr(et, 7, fill = NA)) %>% 
  mutate(et_agg100 = rollsumr(et, 100, fill = NA)) %>% 
  mutate(et_agg365 = rollsumr(et, 365, fill = NA)) %>% 
  mutate(tmax_agg7 = rollmeanr(tx, 7, fill = NA)) %>% 
  mutate(tmin_agg7 = rollmeanr(tn, 7, fill = NA)) %>% 
  mutate(tmean_agg7 = rollmeanr(tg, 7, fill = NA)) %>% 
  mutate(tmax_agg30 = rollmeanr(tx, 30, fill = NA)) %>% 
  mutate(tmin_agg30 = rollmeanr(tn, 30, fill = NA)) %>% 
  mutate(tmean_agg30 = rollmeanr(tg, 30, fill = NA)) %>% 
  mutate(tmax_agg100 = rollmeanr(tx, 100, fill = NA)) %>% 
  mutate(tmin_agg100 = rollmeanr(tn, 100, fill = NA)) %>% 
  mutate(tmean_agg100 = rollmeanr(tg, 100, fill = NA)) %>% 
  mutate(tmax_agg365 = rollmeanr(tx, 365, fill = NA)) %>% 
  mutate(tmin_agg365 = rollmeanr(tn, 365, fill = NA)) %>% 
  mutate(tmean_agg365 = rollmeanr(tg, 365, fill = NA)) %>% 
  mutate(pp_agg7 = rollsumr(pp, 7, fill = NA)) %>%
  mutate(pp_agg30 = rollsumr(pp, 30, fill = NA)) %>%
  mutate(pp_agg365 = rollsumr(pp, 365, fill = NA)) %>%
  mutate(hu_agg7 = rollsumr(hu, 7, fill = NA)) %>%
  mutate(hu_agg30 = rollsumr(hu, 30, fill = NA)) %>%
  mutate(hu_agg365 = rollsumr(hu, 365, fill = NA)) %>%
  mutate(fg_agg7 = rollsumr(fg, 7, fill = NA)) %>%
  mutate(fg_agg30 = rollsumr(fg, 30, fill = NA)) %>%
  mutate(fg_agg365 = rollsumr(fg, 365, fill = NA)) %>%
  mutate(qq_agg7 = rollsumr(qq, 7, fill = NA)) %>%
  mutate(qq_agg30 = rollsumr(qq, 30, fill = NA)) %>%
  mutate(qq_agg365 = rollsumr(qq, 365, fill = NA))

df_head <- read_csv('data/Netherlands/heads.csv', ) 

names(df_head)[1] <- 'Date'

df <- left_join(df_in, df_head)

#####Split dataset training and testing

df_train <- df %>% 
  filter(between(Date, date('2000-01-01'), date('2015-09-10')))

df_test <- df %>% 
  filter(between(Date, date('2016-01-01'), date('2021-12-31')))

#####Define models following tidymodels framework

recipe <-
  recipe(head ~ ., data = df_train) %>% 
  update_role(Date, new_role = "ID")

cores <- parallel::detectCores()

rf_mod <-
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>%
  set_engine("ranger", num.threads = cores) %>%
  set_mode("regression")

rf_wflow <-
  workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(recipe)

#####Tune random forest hyperparameters (mtry and min_n)
#mtry = number of randomly selected predictor min_n)
#min_n = minimal node size

#First split traning dataset to evaluate the models on data not used during training 

eval_set <- validation_time_split(df_train %>% filter(is.na(head) == F), prop = 3/4)

#Run model with different set of parameters and record calibration time

start_time <- Sys.time()

rf_res <-
  rf_wflow %>%
  tune_grid(eval_set,
            grid = 50,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(rmse))

end_time <- Sys.time()

time_cal <- end_time - start_time

#Plot results

autoplot(rf_res)

#Select 1 best paramater value combination to apply to all dataset for prediction

rf_best <-
  rf_res %>%
  select_best(metric = "rmse")

#Select 10 best parameter value combination to apply to all dataset for uncertainty analysis

rf_10best <-
  rf_res %>%
  show_best(metric = "rmse", n = 10)

#####Apply best model for prediction following tidymodels framework

rf_mod <-
  rand_forest(mtry = rf_best$mtry, min_n = rf_best$min_n, trees = 1000) %>%
  set_engine("ranger", num.threads = cores, importance = "impurity") %>%
  set_mode("regression")

rf_wflow <-
  workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(recipe)

rf_fit <-
  rf_wflow %>%
  fit(data = df_train %>% filter(is.na(head) == F))

#Plot variable importance

rf_fit %>% 
  extract_fit_parsnip() %>% 
  vip(num_features = 100)

#Predict all data and calculate RMSE on training datase

df_pred <- df %>% 
  filter(Date >= min(df_train$Date))

df_pred_res <- augment(rf_fit, df_pred) %>%
  mutate(res = head - .pred)

rmse_rf <- sqrt(mean(df_pred_res$res^2, na.rm = T))

#######Plot results

ggplot(df_pred_res) +
  geom_path(aes(Date, .pred, col = 'Sim')) +
  geom_point(aes(Date, head, col = 'Obs')) +
  # geom_path(aes(Date, rr_agg/30+370)) +
  theme_bw() +
  ylab('Hydraulic Head (m.a.s.l.)') +
  xlab('') +
  theme(legend.position = c(0.3, 0.9), legend.box = 'horizontal') 


####Stochastic random forest to (try to) assess prediction uncertainties and
#record uncertainty analysis time

start_time <- Sys.time()

df_stoch <- tibble()

for (i in seq(1, 1000)) {
  
  
  #use the 10 best parameter combinations, 100 times each
  
  vec_mtry <- rep(rf_10best$mtry[1:10], 100)
  
  vec_min_n <- rep(rf_10best$min_n[1:10], 100)
  
  rf_mod <-
    rand_forest(mtry = vec_mtry[i], min_n = vec_min_n[i], trees = 1000) %>%
    set_engine("ranger", num.threads = cores) %>%
    set_mode("regression")
  
  rf_wflow <-
    workflow() %>%
    add_model(rf_mod) %>%
    add_recipe(recipe)
  
  rf_fit <-
    rf_wflow %>%
    fit(data = df_train %>% filter(is.na(head) == F))
  
  #Predict all data 
  
  df_pred <- df %>% 
    filter(Date >= min(df_train$Date))
  
  df_pred_res <- augment(rf_fit, df_pred) %>%
    mutate(res = head - .pred)
  
  ######save outputs
  
  df_stoch <- rbind(df_stoch, df_pred_res)
  
}

write_rds(df_stoch, 'submissions/team_Mirkwood/df_stoch_Netherlands.rds')

end_time <- Sys.time()

time_unc <- end_time - start_time

df_stoch <- read_rds('submissions/team_Mirkwood/df_stoch_Netherlands.rds')

df_stoch_95int <- df_stoch %>% 
  group_by(Date) %>% 
  filter(between(.pred, quantile(.pred, 0.025), quantile(.pred, 0.975))) %>%
  summarise(min_pred = min(.pred),
            max_pred = max(.pred),
            mean_pred = mean(.pred),
            median_pred = median(.pred),
            mean_obs = median(head))


ggplot(df_stoch_95int) +
  geom_ribbon(aes(x = Date, ymin = min_pred, ymax = max_pred, fill = 'Sim'),
              col = 'black', lwd = 0.5) +
  geom_path(aes(x = Date, y = .pred), 
            col = 'red', lwd = 0.5, data = df_pred_res) +
  geom_point(aes(x = Date, y = mean_obs, col = 'Obs'), size = 1) +
  scale_color_manual(values = c('orange'), name = '') +
  scale_fill_manual(values = c('cyan'), name = '') +
  theme_bw() +
  ylab('Hydraulic Head (m.a.s.l.)') +
  xlab('') +
  theme(legend.position = c(0.3, 0.9), legend.box = 'horizontal') 

#####Write model outputs

submission_csv <- read_csv('submissions/team_Mirkwood/submission_form_Netherlands.csv') %>% 
  dplyr::select(Date)

df_out <- df_pred_res %>% 
  left_join(., df_stoch_95int) %>% 
  dplyr::select(Date, .pred, min_pred, max_pred) %>% 
  rename(`Simulated Head` = .pred,
         `95% Lower Bound` = min_pred,
         `95% Upper Bound` = max_pred)

left_join(submission_csv, df_out) %>% 
  write_csv('submissions/team_Mirkwood/submission_form_Netherlands_completed.csv')


