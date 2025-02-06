library(tidyverse)
library(here)
library(mlogit)
library(knitr)
library(caret)

here("code",
     "mlogit_helpers.R") |>
  source()

'%!in%' <- function(x,y)!('%in%'(x,y))

hh_data <- here("data",
                "NHTS",
                "hhpub.csv") |>
  read_csv(show_col_types = FALSE)

person_data <- here("data",
                    "NHTS",
                    "perpub.csv") |>
  read_csv(show_col_types = FALSE)

hh_data <- hh_data |>
  select(WRKCOUNT,
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
         DRIVER)
        
hh_data <- hh_data |>
  mutate(veh_avail = case_when(HHVEHCNT == 0 ~ "Zero",
                               DRVRCNT > HHVEHCNT ~ "Insuff.",
                               TRUE ~ "Suff."))

hh_data <- hh_data |>
  mutate(n_child = HHSIZE - NUMADLT)

n_seniors <- person_data |>
  mutate(is_senior = R_AGE > 64) |>
  group_by(HOUSEID) |>
  summarise(n_seniors = sum(is_senior))

hh_data <- hh_data |>
  left_join(n_seniors)

hh_data <- hh_data |>
  mutate(three_drivers = DRVRCNT > 2)

hh_data <- hh_data |>
  mutate(n_extra_drivers = ifelse(three_drivers, DRVRCNT - 2, 0))

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

non_work_driver <- person_data |>
  mutate(non_work_driver = WORKER == "02" & DRIVER == "01") |>
  group_by(HOUSEID) |>
  summarise(non_work_driver = max(non_work_driver))

hh_data <- hh_data |>
  left_join(non_work_driver)

hh_data <- hh_data |>
  filter(HBPPOPDN > 0) |>
  mutate(density = case_when(HBPPOPDN < 7000 ~ "Low",
                             HBPPOPDN < 10000 ~ "High",
                             TRUE ~ "Medium"))

hh_data <- hh_data |>
  select(HOUSEID,
         veh_avail,
         WRKCOUNT,
         n_child,
         n_seniors,
         n_extra_drivers,
         three_drivers,
         non_work_driver,
         income,
         density)

set.seed(3775668)

hh_data_train_ids <- sample(hh_data$HOUSEID, 
                        size = ceiling(nrow(hh_data)/2))

hh_data_train <- hh_data |>
  filter(HOUSEID %in% hh_data_train_ids)

hh_data_test <- hh_data |>
  filter(HOUSEID %!in% hh_data_train_ids)

veh_dfidx_train <- fn_make_dfidx(hh_data_train,
                                "HOUSEID",
                                "veh_avail")

veh_dfidx_test <- fn_make_dfidx(hh_data_test,
                                "HOUSEID",
                                "veh_avail")

model_veh <- mlogit(choice ~ 0 | 
                      WRKCOUNT +
                      n_child +
                      n_seniors +
                      n_extra_drivers +
                      three_drivers + 
                      non_work_driver +
                      income +
                      density | 0,
                           veh_dfidx_train,
                           reflevel = "Suff.")

summary(model_veh)