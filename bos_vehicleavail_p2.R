# MEMO: what are the predictors of hh veh avail
# how do my results compare given sample (local vs natnl)

library(tidyverse)
library(here)
library(mlogit)
library(knitr)
library(caret)

#import functions from mlogit_helpers.R
here("code", "mlogit_helpers.R") |> source()

#select rows based on ID not list
"%!in%" <- function(x, y) ! ("%in%"(x, y))

#load datasets
#hh is household
hh_data <- here("data", "NHTS", "hhpub.csv") |>
  read_csv(show_col_types = FALSE)
person_data <- here("data", "NHTS", "perpub.csv") |>
  read_csv(show_col_types = FALSE)
trip_data <- here("data", "NHTS", "trippub.csv") |>
  read_csv(show_col_types = FALSE)
veh_data <- here("data", "NHTS", "vehpub.csv") |>
  read_csv(show_col_types = FALSE)

#select columns from hh and person datasets
hh_data <- hh_data |>
  select(WRKCOUNT,
         CAR,
         DRVRCNT,
         HHVEHCNT,
         HHSIZE,
         NUMADLT,
         HHFAMINC,
         HBPPOPDN,
         HOUSEID)

person_data <- person_data |>
  select(HOUSEID,
         R_AGE,
         WORKER,
         DRIVER,
         R_RELAT)


#assign categorical outcome of vehicle availability
hh_data <- hh_data |>
  mutate(veh_avail = case_when(HHVEHCNT == 0 ~ "Zero",
                               DRVRCNT > HHVEHCNT ~ "Insuff.",
                               TRUE ~ "Suff."))

#get number of children
hh_data <- hh_data |>
  mutate(n_child = HHSIZE - NUMADLT)

#get number of seniors
n_seniors <- person_data |>
  mutate(is_senior = R_AGE > 64) |>
  group_by(HOUSEID) |>
  summarise(n_seniors = sum(is_senior))

hh_data <- hh_data |>
  left_join(n_seniors)

#get number of partner
n_partner <- person_data |>
  mutate(has_partner = R_RELAT == "02") |>
  group_by(HOUSEID) |>
  summarise(n_partner = sum(has_partner))

hh_data <- hh_data |>
  left_join(n_partner)

#get presence of >2 drivers (t/f)
hh_data <- hh_data |>
  mutate(three_drivers = DRVRCNT > 2)

#get number of addit drivers over 2
hh_data <- hh_data |>
  mutate(n_extra_drivers = ifelse(three_drivers, DRVRCNT - 2, 0))

#get income level (low, medium, high)
#TODO: consider changing income to continuous and log-transform it
hh_data <- hh_data |>
  mutate(HHFAMINC = as.numeric(HHFAMINC)) |>
  filter(HHFAMINC > 0) |>
  mutate(income = case_when(HHFAMINC < 4 ~ "low",
                            HHFAMINC < 5 & HHSIZE > 1 ~ "low",
                            HHFAMINC < 6 & HHSIZE > 3 ~ "low",
                            HHFAMINC < 7 & HHSIZE > 5 ~ "low",
                            HHFAMINC < 8 & HHSIZE > 7 ~ "low",
                            HHFAMINC > 8 ~ "high",
                            TRUE ~ "medium")) |>
  mutate(income = factor(income, levels = c("medium", "low", "high")))

#get households' status: if they have a non-worker driver or not
#adds non_work_driver column (true/false) to hh_data
#see codebook for defns of "01" and "02"
non_work_driver <- person_data |>
  mutate(non_work_driver = WORKER == "02" & DRIVER == "01") |>
  group_by(HOUSEID) |>
  summarise(non_work_driver = max(non_work_driver))

hh_data <- hh_data |>
  left_join(non_work_driver)

#categorical vehicle use freq
hh_data <- hh_data |>
  mutate(car_freq = case_when(CAR == "01" ~ "Daily",
                              CAR == "02" ~ "Semi-weekly",
                              CAR == "03" ~ "Semi-monthly",
                              CAR == "04" ~ "Semi-yearly",
                              CAR == "05" ~ "Never"))

#make new column with "density" categorical var based on pop density
hh_data <- hh_data |>
  filter(HBPPOPDN > 0) |>
  mutate(density = case_when(HBPPOPDN < 7000 ~ "Low",
                             HBPPOPDN < 10000 ~ "High",
                             TRUE ~ "Medium"))

#cleaning dataset
#keep only computed variables and those needed for model
hh_data <- hh_data |>
  select(HOUSEID,
         veh_avail,
         WRKCOUNT,
         car_freq,
         n_child,
         n_seniors,
         n_partner,
         n_extra_drivers,
         three_drivers,
         non_work_driver,
         income,
         density)

#set seed
set.seed(1312025)

#new object with half the hh set for training
hh_data_train_ids <- sample(hh_data$HOUSEID,
                            size = ceiling(nrow(hh_data) / 2))

hh_data_train <- hh_data |>
  filter(HOUSEID %in% hh_data_train_ids)

#the other half of hh for testing
hh_data_test <- hh_data |>
  filter(HOUSEID %!in% hh_data_train_ids)

#reformat to use dfidx format for multinominal logistic regression
#dfidx training set
veh_dfidx_train <- fn_make_dfidx(hh_data_train,
                                "HOUSEID",
                                "veh_avail")

#dfidx testing set
veh_dfidx_test <- fn_make_dfidx(hh_data_test,
                                "HOUSEID",
                                "veh_avail")

#estimate multinominal logistic regression
model_veh <- mlogit(choice ~ 0 |
                    WRKCOUNT +
                    car_freq +
                    n_child +
                    n_seniors +
                    n_partner +
                    n_extra_drivers +
                    three_drivers + 
                    non_work_driver +
                    income +
                    density | 0,
                    veh_dfidx_train,
                    reflevel = "Suff.")

summary(model_veh)

predicts_test <- predict(model_veh, veh_dfidx_test) |>
  as.data.frame() |>
  rownames_to_column("HOUSEID") |>
  mutate(HOUSEID = as.numeric(HOUSEID)) |>
  left_join(hh_data_test)

head(predicts_test) |>
  kable()

predicts_test <- predicts_test |>
  mutate(most_likely = case_when((Suff. > Insuff.) & (Suff. > Zero) ~ "Suff.",
                                 (Zero > Insuff.) & (Zero > Suff.) ~ "Zero",
                                 TRUE ~ "Insuff.")) 

predicts_test <- predicts_test |>
  mutate(most_likely = factor(most_likely, 
                              levels = c("Suff.", "Insuff.", "Zero"))) |>
  mutate(veh_avail = factor(veh_avail,
                            levels = c("Suff.", "Insuff.", "Zero"))) |>
  mutate(correct = veh_avail == most_likely)

confusionMatrix(data = predicts_test$most_likely,
                reference = predicts_test$veh_avail)
