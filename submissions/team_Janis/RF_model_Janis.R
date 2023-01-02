# R version 4.2.2
library(tidymodels) # version 1.0.0
library(tidyverse) # version 1.3.2
library(lubridate) # version 1.9.0
library(timetk) # version 2.8.2
library(ranger) # version 0.14.1

# slidify functions -------------------------------------------------------
roll_mean_5 <- slidify(mean, .period = 5, .align = "right", .partial = TRUE)
roll_mean_30 <- slidify(mean, .period = 30, .align = "right", .partial = TRUE)
roll_mean_60 <- slidify(mean, .period = 60, .align = "right", .partial = TRUE)
roll_mean_180 <- slidify(mean, .period = 180, .align = "right", .partial = TRUE)
roll_sum_5 <- slidify(sum, .period = 5, .align = "right", .partial = TRUE)
roll_sum_30 <- slidify(sum, .period = 30, .align = "right", .partial = TRUE)
roll_sum_60 <- slidify(sum, .period = 60, .align = "right", .partial = TRUE)
roll_sum_180 <- slidify(sum, .period = 180, .align = "right", .partial = TRUE)
roll_sum_270 <- slidify(sum, .period = 270, .align = "right", .partial = TRUE)
roll_sum_365 <- slidify(sum, .period = 365, .align = "right", .partial = TRUE)

# Input prepare function -----------------------------------------------------
prepare_features <- function(df) {
  df_ <- df %>%
    rename_with(.fn = ~paste0(., "_"), .cols = matches(c("tg", "tn", "tx", "pp", "fg"))) %>% # add '_' at the end of original features
    mutate(across(ends_with("_"), roll_mean_5, .names = "{col}mean_5")) %>%
    mutate(across(ends_with("_"), roll_mean_30, .names = "{col}mean_30")) %>%
    mutate(across(ends_with("_"), roll_mean_60, .names = "{col}mean_60")) %>%
    mutate(across(ends_with("_"), roll_mean_180, .names = "{col}mean_180")) %>%
    mutate(inf_delta_0 = rr - et) %>% # delta infiltration
    mutate(infabs = if_else(inf_delta_0 > 0, true = inf_delta_0, false = 0)) %>%
    mutate(tg_positives = if_else(tg_ >0, true = tg_, false = 0)) %>% # to calculate a sum of non-freezing temperatures
    mutate(across(c("et", "rr", "qq", "infabs", "tg_positives"), roll_sum_5, .names = "{col}_sum_5")) %>%
    mutate(across(c("et", "rr", "qq", "infabs", "tg_positives"), roll_sum_30, .names = "{col}_sum_30")) %>%
    mutate(across(c("et", "rr", "qq", "infabs", "tg_positives"), roll_sum_60, .names = "{col}_sum_60")) %>%
    mutate(across(c("et", "rr", "qq", "infabs", "tg_positives"), roll_sum_180, .names = "{col}_sum_180")) %>%
    mutate(across(c("et", "rr", "qq", "infabs", "tg_positives"), roll_sum_270, .names = "{col}_sum_270")) %>%
    mutate(inf_delta_5 = rr_sum_5 - et_sum_5) %>% 
    mutate(inf_delta_30 = rr_sum_30 - et_sum_30) %>% 
    mutate(inf_delta_60 = rr_sum_60 - et_sum_60) %>% 
    mutate(inf_delta_180 = rr_sum_180 - et_sum_180) %>% 
    mutate(inf_delta_270 = rr_sum_270 - et_sum_270) %>% 
    mutate(inf_ratio_30 = rr_sum_30 / et_sum_30) %>% # P/ET ratio 
    mutate(inf_ratio_60 = rr_sum_60 / et_sum_60) %>%
    mutate(inf_ratio_180 = rr_sum_180 / et_sum_180) %>%
    mutate(inf_ratio_270 = rr_sum_270 / et_sum_270) 
  return(df_)
}
# Input prepare function for USA  --------------------------------------------------
prepare_features_USA <- function(df) {
  df_ <- df %>%
    rename_with(.fn = ~paste0(., "_"), .cols = matches(c("tg", "tn", "tx", "Stage_m"))) %>% 
    mutate(across(ends_with("_"), roll_mean_5, .names = "{col}mean_5")) %>%
    mutate(across(ends_with("_"), roll_mean_30, .names = "{col}mean_30")) %>%
    mutate(across(ends_with("_"), roll_mean_60, .names = "{col}mean_60")) %>%
    mutate(across(ends_with("_"), roll_mean_180, .names = "{col}mean_180")) %>%
    mutate(inf_delta_0 = rr - et) %>% 
    mutate(infabs = if_else(inf_delta_0 > 0, true = inf_delta_0, false = 0)) %>%
    mutate(tg_positives = if_else(tg_ >0, true = tg_, false = 0)) %>%
    mutate(across(c("et", "rr", "infabs", "tg_positives"), roll_sum_5, .names = "{col}_sum_5")) %>%
    mutate(across(c("et", "rr", "infabs", "tg_positives"), roll_sum_30, .names = "{col}_sum_30")) %>%
    mutate(across(c("et", "rr", "infabs", "tg_positives"), roll_sum_60, .names = "{col}_sum_60")) %>%
    mutate(across(c("et", "rr", "infabs", "tg_positives"), roll_sum_180, .names = "{col}_sum_180")) %>%
    mutate(across(c("et", "rr", "infabs", "tg_positives"), roll_sum_270, .names = "{col}_sum_270")) %>%
    mutate(inf_delta_5 = rr_sum_5 - et_sum_5) %>% 
    mutate(inf_delta_30 = rr_sum_30 - et_sum_30) %>% 
    mutate(inf_delta_60 = rr_sum_60 - et_sum_60) %>% 
    mutate(inf_delta_180 = rr_sum_180 - et_sum_180) %>% 
    mutate(inf_delta_270 = rr_sum_270 - et_sum_270) %>% 
    mutate(inf_ratio_30 = rr_sum_30 / et_sum_30) %>%
    mutate(inf_ratio_60 = rr_sum_60 / et_sum_60) %>%
    mutate(inf_ratio_180 = rr_sum_180 / et_sum_180) %>%
    mutate(inf_ratio_270 = rr_sum_270 / et_sum_270) 
  return(df_)
}

# function to prepare resamples --------------------------------------------------
# it makes smaller chunks of test/train datasets 
# using resamples by moving 2 consecutive (calendar) year window
prepare_resamples <- function(df_input) {
  df_ <- df_input %>%
    mutate(.row = row_number()) %>%
    group_by(grupa = floor_date(Date, unit = '2 years')) %>%
    mutate(split = cur_group_id()) %>%
    ungroup()
  
  all_splits <- unique(df_$split)
  
  make_prepared_list <- function(df, split) {
    selected_split <- df$split == split
    return(list(analysis = df$.row[!selected_split], assessment = df$.row[selected_split]))
  }
  
  prepared_list <- purrr::map(all_splits, ~make_prepared_list(df_, .x))
  splits_ready <- lapply(prepared_list, make_splits, data = df_)
  resamples_ready <- manual_rset(splits_ready, ids = as.character(all_splits))
  return(resamples_ready)
}



# Define locations and train/test dates-----------------------------------------
info <- list(tibble(well = "Germany", train_start = as_date("2002-05-01"), train_end = as_date("2016-12-31"),
            test_start = as_date("2017-01-01"), test_end = as_date("2022-12-31")),
     tibble(well = "Netherlands", train_start = as_date("2000-01-01"), train_end = as_date("2015-09-10"),
            test_start = as_date("2016-01-01"), test_end = as_date("2021-12-31")),
     tibble(well = "Sweden_1", train_start = as_date("2001-01-01"), train_end = as_date("2015-12-31"),
            test_start = as_date("2016-01-01"), test_end = as_date("2021-12-31")),
     tibble(well = "Sweden_2", train_start = as_date("2001-01-01"), train_end = as_date("2015-12-31"),
            test_start = as_date("2016-01-01"), test_end = as_date("2021-12-31")),
     tibble(well = "USA", train_start = as_date("2002-03-01"), train_end = as_date("2016-12-31"),
     test_start = as_date("2017-01-01"), test_end = as_date("2022-05-31")))


# to save calibration times
cal_times <- list()

set.seed(123)

for (i in 1:5) {
  
  print(info[[i]])
  
  # import data -------------------------------------------------------------
  # Working directory is the upper level of "challenge" repository
  
  heads <- read_csv(paste0("data/", info[[i]]$well, "/heads.csv")) %>%
    rename(Date = 1) # because of the Netherlands well
  
  # Different data preparation approach for USA
  if(info[[i]]$well == "USA") {
    input_full <- read_csv("data/USA/input_data.csv") %>%
      rename(Date = 1, rr = PRCP, tx = TMAX, tm = TMIN, et = ET) %>%
      mutate(tg = (tx+tm)/2)
  } else {
    input_full <- read_csv(paste0("data/", info[[i]]$well, "/input_data.csv")) %>%
      rename(Date = time)
  }
  
  # remove testing period:
  input_sel <- input_full %>%
    filter(Date <= info[[i]]$train_end)
  
  
  # prepare inputs ----------------------------------------------------------
  
  # For USA different function due to other variables
  if(info[[i]]$well == "USA") {
    input_selprep <- prepare_features_USA(input_sel)
  } else {
    input_selprep <- prepare_features(input_sel)
  }
  
  # join head data and leave only entries with head values
  input <- input_selprep %>%
    left_join(heads, by = "Date") %>%
    filter(!is.na(head))
    
  # prepare resamples having 2-year moving window
  resamples <- prepare_resamples(input)
  
  
  # create a model ----------------------------------------------------------
  
  rand_spec <- rand_forest(
    mtry  = tune(), # tuning only mtry 
    trees  = 500, 
    min_n = 1
  ) %>% 
    set_engine("ranger", 
               num.threads = 8,
               importance = "none", # for feature testing experiments "impurity" was used
               keep.inbag=TRUE
               ) %>% 
    set_mode("regression")
  
  # Recipe, although most of new features were created by "prepare_features" function
  rf_recipe <- recipe(head ~ ., data = input) %>% 
    step_date(Date, features = c("doy", "month"),
              keep_original_cols = FALSE)
  
  # combine in a workflow
  rand_wf <- workflow() %>%
    add_recipe(rf_recipe) %>%
    add_model(rand_spec)
  
  # simple tuning grid for mtry
  rf_grid <- expand_grid(mtry = c(5, 10, 15, 20, 25, 30, 40))
  
  { # tuning mtry using resamples
    t <- Sys.time()
  
    rf_res <- tune_grid(
      rand_wf,
      resamples = resamples, # using splits of 2-year windows
      grid = rf_grid
    )
    cal_time <- as.numeric(difftime(Sys.time(), t, units = "secs"))
  }
  
  # select model with the best performing mtry 
  best_rf <- select_best(rf_res, metric = "rmse")
  final_rf_workflow <- rand_wf %>%
    finalize_workflow(best_rf) 
  
  final_model <- final_rf_workflow %>%
    fit(data = input)
  
  # Testing part ---------------------------------------------------------
  
  
  if(info[[i]]$well == "USA") {
    input_pred <- prepare_features_USA(input_full) %>%
      filter(Date >= info[[i]]$train_start & Date <= info[[i]]$test_end)
  } else {
    input_pred <- prepare_features(input_full) %>%
      filter(Date >= info[[i]]$train_start & Date <= info[[i]]$test_end)
  }

  # final results
  res <- input_pred %>%
    mutate(pred_conf_intervals = predict(final_model, new_data = .,type = 'conf_int', level = 0.95),
           pred = predict(final_model, new_data = .)) %>%
    select(Date, pred, pred_conf_intervals) %>% 
    unnest(c(pred, pred_conf_intervals)) %>%
    rename("Simulated Head" = .pred,
           "95% Lower Bound" = .pred_lower,
           "95% Upper Bound" = .pred_upper)
  
  write_csv(res, paste0("submissions/team_Janis/submission_from_", info[[i]]$well, ".csv"))
  
  # saving calibration time:
  cal_times[[info[[i]]$well]] <- cal_time
}

