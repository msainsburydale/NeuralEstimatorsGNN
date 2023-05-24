library("NeuralEstimators")
library("ggplot2")
library("dplyr")
library("ggpubr")
library("viridis")
library("tidyr")


# Legend labelling 
estimator_order <- names(estimator_labels) # specifies the order that the estimators should appear in the plot legends.
scale_estimator <- function(df, scale = "colour", values = estimator_colours, ...) {
  estimators <- unique(df$estimator)
  ggplot2:::manual_scale(
    scale,
    values = values[estimators],
    labels = estimator_labels,
    breaks = estimator_order,
    ...
  )
}

# TODO move asterix to the left (maybe with negative spacing?)
estimator_labels <- c(
  "GNN1" = expression(hat(theta)("·"~";"~gamma[30]*"*")),
  "GNN2" = expression(hat(theta)("·"~";"~gamma[300]*"*")),
  "GNN3" = expression(hat(theta)("·"~";"~gamma[30:300]*"*")),
  "GNN" = "GNN",
  "CNN" = "CNN",
  "MAP" = "MAP"
)

estimator_colours <- c(
  "MAP" = "#FDE725FF",
  "GNN" = "#21908CFF",
  "CNN" = "#440154FF",
  # Variable sample size experiment:
  "GNN1"    = "red",
  "GNN2"    = "orange",
  "GNN3"    = "#440154FF"
  
)



# TODO update plotdistribution() to automatically select only the parameters that are in df, rather than throwing an error when "The number of parameter labels differs to the number of parameters"
parameter_labels <- c(
  "τ"  = expression(hat(tau)),
  # "σ"  = expression(sigma), 
  "ρ"  = expression(hat(rho))
)

loss <- function(x, y) abs(x - y)

# The simulations may be highly varied in magnitude, so we need to
# use an independent colour scale. This means that we can't use facet_wrap().
field_plot <- function(field, regular = TRUE, variable = "Z", x = "s1", y = "s2") {
  
  # Standard eval with ggplot2 without `aes_string()`: https://stackoverflow.com/a/55133909
  
  gg <- ggplot(field, aes(x = !!sym(x), y = !!sym(y)))
  
  if (regular) {
    gg <- gg +
      geom_tile(aes(fill = !!sym(variable))) +
      scale_fill_viridis_c(option = "magma")
  } else {
    gg <- gg +
      geom_point(aes(colour = !!sym(variable))) +
      scale_colour_viridis_c(option = "magma")
  }
  
  gg <- gg +
    labs(fill = "", x = expression(s[1]), y = expression(s[2])) +
    theme_bw() +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0))
  
  return(gg)
}



MAE <- function(x, y) mean(abs(x - y))
MSE <- function(x, y) mean((x - y)^2)
RMSE <- function(x, y) sqrt(mean((x - y)^2))
MAD <- mad
zeroone <- function(x, y, eps = y/10) mean(abs(x - y) > eps)

# TODO determine a way to transfer the facet labels to the title
splitfacet <- function(x){
  
  facet_vars <- names(x$facet$params$facets)         # get the names of the variables used for faceting
  x$facet    <- ggplot2::ggplot()$facet              # overwrite the facet element of our plot object with the one from the empty ggplot object (so if we print it at this stage facets are gone)
  datasets   <- split(x$data, x$data[facet_vars])    # extract the data and split it along the faceting variables
  new_plots  <- lapply(datasets,function(new_data) { # overwrite the original data with each subset and store all outputs in a list
    x$data <- new_data
    x})
}


risklabel <- expression(r[Omega](hat(theta)("·")))
