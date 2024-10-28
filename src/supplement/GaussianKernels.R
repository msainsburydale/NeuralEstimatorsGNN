# Load necessary libraries
library(ggplot2)
library(dplyr)
library(gridExtra)

# Define the intervals for the original indicator functions
intervals <- data.frame(
  a = seq(0, 0.135, by = 0.015),
  b = seq(0.015, 0.15, by = 0.015),
  mid = (seq(0, 0.135, by = 0.015) + seq(0.015, 0.15, by = 0.015)) / 2
)

# Define the Gaussian kernel approximation
gaussian_kernel <- function(d, mu, sigma) {
  exp(-(d - mu)^2 / (2 * sigma^2))
}

# Create a sequence of distances (d)
d_vals <- seq(0, 0.15, length.out = 1000)

# Define means and standard deviations for the Gaussian kernels
mu_vals <- intervals$mid
sigma_vals <- rep(0.00375, length(mu_vals))

# Create a data frame for Gaussian kernel approximations
gaussian_df <- data.frame(
  d = rep(d_vals, times = length(mu_vals)),
  value = unlist(lapply(1:length(mu_vals), function(i) gaussian_kernel(d_vals, mu_vals[i], sigma_vals[i]))),
  label = factor(rep(paste0("Gaussian ", 1:length(mu_vals)), each = length(d_vals)))
)

# Create a data frame for the original indicator functions
indicator_df <- data.frame(
  d = rep(d_vals, times = length(mu_vals)),
  value = unlist(lapply(1:length(mu_vals), function(i) ifelse(d_vals >= intervals$a[i] & d_vals <= intervals$b[i], 1, 0))),
  label = factor(rep(paste0("Indicator ", 1:length(mu_vals)), each = length(d_vals)))
)

# Plot the indicator functions
p1 <- ggplot(indicator_df, aes(x = d, y = value, group = label, fill = label)) +
  geom_area(alpha = 0.5) +
  ggtitle("Indicator Functions") +
  labs(x = "distance", y = "Value") +
  theme_minimal() + 
  theme(legend.position = "none") 

# Plot the Gaussian kernel approximations
p2 <- ggplot(gaussian_df, aes(x = d, y = value, group = label, color = label)) +
  geom_line(size = 1) +
  ggtitle("Gaussian Kernel Approximations") +
  labs(x = "distance", y = "Value") +
  theme_minimal() + 
  theme(legend.position = "none") 

# Combine and display the plots
grid.arrange(p1, p2, ncol = 1)
