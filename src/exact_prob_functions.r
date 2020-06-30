# quantile to points

#' @param quantiles series the quantile outputs after using predict on a `rq` model
#'
#' @return list of all the points presenting that quantile distribution

quantiles_to_points <- function(quantiles) {
  
  # Creating a empty vector to store all the points that will represent the quantile distribution
  my_points <- vector('double', length(quantiles))
  
  for (n in 1:(length(quantiles) + 1)) {
    
    if (n == 1) {
      # This deals with the first quantile point
      quantile_flow <- quantiles[1] - 0.5*(quantiles[2] - quantiles[1])
      my_points[n] <- quantile_flow
      
    } else if (n == (length(quantiles) + 1)) {
      # This deals with the last quantile point
      quantile_flow <- quantiles[n-1] + 0.5*(quantiles[n-1] - quantiles[n-2])
      my_points[n] <- quantile_flow
      
    } else {
      # Quantiles in the middle
      quantile_flow <- (quantiles[n] + quantiles[n-1])/2
      my_points[n] <- quantile_flow
      
    }
  }
  return(my_points)
}


# condensing sampled points

#' @param temp_data list of all data points generated from the different input data sets used
#' @param resolution int how many data points are generated from a single input. For example, if our model generates quantiels 1:99/100, then the resolution would be 100, as it is the number of points that can be generated to represent the 99 quantiles.
#'
#' @return list a condensed list of all of the data points we have generated

condense_points <- function(temp_data, resolution) {
  # Seeing how big each of the bins should be
  bin_size <- length(temp_data)/resolution
  my_sorted <- sort(temp_data)
  condensed_list <- vector('double', resolution)
  
  for (n in 1:resolution) {
    # taking the mean of each of the bins
    condensed_list[n] <- mean(my_sorted[((n - 1)*bin_size + 1):(n*bin_size)])
  }
  
  return(condensed_list)
}

# creating new sample points

#' @param current_day int the number of days that we have already simulated
#' @param resolution int how many data points are generated from a single input. For example, if our model generates quantiels 1:99/100, then the resolution would be 100, as it is the number of points that can be generated to represent the 99 quantiles.
#' @param max_lags int the maximum number of autoregressive lags that our model looks at
#' @param input_data data.frame Data Frame of unique input sets we currently have
#' @param sample_storage data.frame Data Frame containing the unique sample points for each individual day
#'
#' @return list a condensed list of all of the data points we have generated

create_input_data <- function(current_day, resolution, max_lags, input_data, sample_storage) {
  
  # Seeing how many rows are required to store all possible combinations
  if (current_day <= max_lags) {
    num_rows <- resolution**current_day
  } else {
    num_rows <- resolution**max_lags
  }
  
  # Creating empty DataFrame to store input sets
  data_storage <- data.frame(matrix(ncol=max_lags, nrow=num_rows))
  
  if (current_day < max_lags) {
    # Copying data over that remains the same
    data_storage[, (current_day + 1):max_lags] <- input_data[, current_day:(max_lags - 1)]
    # Data that needs to be permutated
    data_storage[, 1:current_day] <- expand.grid(sample_storage[, current_day:1])
  } else {
    data_storage[, 1:max_lags] <- expand.grid(sample_storage[, current_day:(current_day-max_lags+1)])
  }
  
  colnames(data_storage) <- c('lag1', 'lag2', 'lag3')
  
  return(data_storage)
}

