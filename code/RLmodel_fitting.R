# Load necessary packages
if (!requireNamespace("Matrix", quietly = TRUE))
    install.packages("Matrix")
library(Matrix)

softmax <- function(Q_values, tau) {
  exp(tau*Q_values) / sum(exp(tau*Q_values))
}

neg_log_likelihood <- function(data, parameters) {
  tau <- parameters[1] # decision temperature
  lambda <- parameters[2] # learning rate
  LL <- 0

  Q_values <- matrix(0.5, nrow = nrow(data), ncol = max(data$choice))
  choice_probabilities <- matrix(NA, nrow = nrow(data), ncol = max(data$choice))

  for (t in 1:nrow(data)) {
    choice_probabilities[t,] <- softmax(Q_values[t,], tau)
    if (t < nrow(data)) {
      Q_values[t+1,] <- Q_values[t,]
      Q_values[t+1,data$choice[t]] <- Q_values[t,data$choice[t]] + lambda * (data$reward[t] - Q_values[t,data$choice[t]])
      LL <- LL + log(choice_probabilities[t, data$choice[t]])
    }
  }

  return(LL)
}
