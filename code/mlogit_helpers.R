## Libraries
library(here)
library(tidyverse)
library(mlogit)

# Return the most frequent category
freq_cat <- function(x) {
  uniqx <- unique(na.omit(x))
  uniqx[which.max(tabulate(match(x, uniqx)))]
}

## A function to create a dfidx object from an individual-level 
## dataset (no alternative-level variables)
fn_make_dfidx <- function(my_situation, 
                          my_id, 
                          my_alts) {
  
  options <- base::unique(my_situation[[my_alts]])
  
  base::colnames(my_situation)[colnames(my_situation) == my_alts] ="alternatives"
  base::colnames(my_situation)[colnames(my_situation) == my_id] ="id"
  
  rep_trips <- my_situation |>
    dplyr::mutate(avail_choice = options[1]) 
  
  for (i in 2:length(options)){
    next_trips <- my_situation |>
      dplyr::mutate(avail_choice = options[i])
    rep_trips = dplyr::bind_rows(rep_trips, next_trips)
  }
  
  rep_trips <- rep_trips |>
    dplyr::arrange(id) |>
    dplyr::mutate(choice = ifelse(alternatives == avail_choice, TRUE, FALSE)) |> 
    dplyr::relocate(choice) |>
    dplyr::relocate(avail_choice) |>
    dplyr::relocate(id) |>
    dplyr::select(-alternatives) 
  
  dfidx <- dfidx::dfidx(rep_trips, drop.index = FALSE)
}

## A function to make predictions from an mlogit model varying
## one variable and holding the others (mostly) constant at their means
fn_predictions <- function(my_model,
                           to_vary,
                           max_sample) {
  
  # Get list of continuous variables to hold constant
  to_hold_names <- 
    names(my_model$model)[2:(length(names(my_model$model))-3)]
  
  orig_data <- tibble(my_model$model$idx[1]) 
  names(orig_data) <- "person"
  
  for (i in 1:length(to_hold_names)) {
    if(grepl("log\\(", to_hold_names[i])) {
      orig_data[, substr(to_hold_names[i], 
                         5, 
                         str_length(to_hold_names[i])-1)] <- 
        exp(my_model$model[[to_hold_names[i]]])
    } else {
      orig_data[, to_hold_names[i]] <- my_model$model[[to_hold_names[i]]]
    }
  }
  
  orig_data <- orig_data |>
    na.omit() |>
    group_by(person) |>
    summarise(across(everything(), ~ first(.x))) |>
    select(-person)
  
  orig_cont <- orig_data[sapply(orig_data, is.numeric)]
  orig_cat <- orig_data[!sapply(orig_data, is.numeric)]
  
  to_hold_cont <- sapply(orig_cont, mean)[names(orig_cont) != to_vary]
  to_hold_cat <- sapply(orig_cat, freq_cat)[names(orig_cat) != to_vary]
  
  # Make a list of alternatives
  options <- unique(my_model$model$idx[[2]])
  
  # Set up a data frame to put constant/varying values in.
  my_data <- tibble(id = seq(1, max_sample, by = 1),
                    choice = sample(options, 
                                    size = max_sample, replace = TRUE))
  
  # In case the variable to vary is log-transformed
  if(sum(names(my_model$model) == to_vary) == 1) {
    vary_data <- my_model$model[which(names(my_model$model) == to_vary)][[1]]
  } else if (sum(names(my_model$model) == paste0("log(",to_vary,")")) == 1) {
    vary_data <- exp(my_model$model[which(names(my_model$model) == 
                                            paste0("log(",to_vary,")"))][[1]])
  }
  
  
  if(is.numeric(vary_data)) {
    # For continuous variables, vary the data across its range from the model
    vary_limits <- c(min(vary_data, na.rm = TRUE),
                     max(vary_data, na.rm = TRUE))
    
    my_data[, to_vary] <- seq(min(vary_limits) + 
                                ((max(vary_limits) - min(vary_limits)) / max_sample), 
                              max(vary_limits), 
                              (max(vary_limits) - min(vary_limits)) / max_sample)
  } else {
    # For discrete variables, randomly sample values.
    my_data[, to_vary] <- sample(na.omit(unique(my_model$model[[to_vary]])),
                                 size = max_sample,
                                 replace = TRUE)
  }
  
  for(i in 1:length(to_hold_cont)) {
    my_data[, names(to_hold_cont)[i]] <- rnorm(max_sample, 
                                               mean = to_hold_cont[i], 
                                               sd = to_hold_cont[i]/10000)
  }
  
  for(i in 1:length(to_hold_cat)) {
    my_data[, names(to_hold_cat)[i]] <- 
      sample(na.omit(unique(my_model$model[[names(to_hold_cat)[i]]])), 
             size = max_sample,
             replace = TRUE)
  }
  
  
  my_dfidx <- fn_make_dfidx(my_data,
                            "id",
                            "choice")
  
  my_preds <- predict(my_model, my_dfidx) |>
    cbind(my_data) 
  
  for(i in 1:length(to_hold_cat)) {
    my_preds <- my_preds[my_preds[[names(to_hold_cat)[i]]] == to_hold_cat[i],]
  } 
  
  my_preds <- my_preds[colnames(my_preds) %in% options | 
                         colnames(my_preds) == to_vary] |>
    pivot_longer(cols = !contains(to_vary),
                 names_to = "alternative",
                 values_to = "probability") 
  
  if(!is.numeric(my_data[[to_vary]])) {
    names(my_preds)[1] <- "category"
    
    my_preds <- my_preds |>
      group_by(category, alternative) |>
      summarise(probability = mean(probability))
  }
  
  names(my_preds)[1] <- "variable"
  my_preds
}
