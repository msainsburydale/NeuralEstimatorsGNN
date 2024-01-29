suppressMessages({
  library("NeuralEstimators")
  library("ggplot2")
  library("dplyr")
  library("ggpubr")
  library("egg")
  library("viridis")
  library("tidyr")
  library("latex2exp")
  library("ggExtra") # ggMarginal
  library("cowplot") # get_x_axis
  options(dplyr.summarise.inform = FALSE) 
})

if(!interactive()) pdf(NULL)


# see: latex2exp_supported()
estimator_labels <- c(
  # Main text experiments: 
  "GNN" = "GNN",
  "ML" = "ML",
  # Variable sample size experiment:
  "GNN1" = TeX("$\\hat{\\theta}(\\cdot; \\, \\gamma^{*}_{30})$"),
  "GNN2" = TeX("$\\hat{\\theta}(\\cdot; \\, \\gamma^{*}_{1000})$"),
  "GNN3" = TeX("$\\hat{\\theta}(\\cdot; \\, \\gamma^{*}_{30;1000})$"),
  # Neighbourhood experiment: 
  "fixedradius" = "Disc of fixed radius",
  "knearest" = "k-nearest neighbours",
  "combined" = "Random-k neighbours\nwithin disc of fixed radius"
)

estimators <- names(estimator_labels)

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

# for more colours, see: http://www.stat.columbia.edu/~tzheng/files/Rcolor.pdf
estimator_colours <- c(
  # Main text experiments: 
  "ML" = "gold",
  "GNN" = "chartreuse4",
  # Variable sample size experiment:
  "GNN1" = "red",
  "GNN2" = "dodgerblue4",
  "GNN3" = "#440154FF",
  # Neighbourhood experiment: 
  "fixedradius" = "gold",
  "knearest" = "dodgerblue4",
  "combined" = "chartreuse4"
)

parameter_labels <- c(
  "τ"  = expression(hat(tau)),
  "σ"  = expression(hat(sigma)),
  "ρ"  = expression(hat(rho)),
  "ν"  = expression(hat(nu))
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


splitfacet <- function(x){

  # NB For this function be useful, need to transfer the facet labels to the title

  facet_vars <- names(x$facet$params$facets)         # get the names of the variables used for faceting
  x$facet    <- ggplot2::ggplot()$facet              # overwrite the facet element of our plot object with the one from the empty ggplot object (so if we print it at this stage facets are gone)
  datasets   <- split(x$data, x$data[facet_vars])    # extract the data and split it along the faceting variables
  new_plots  <- lapply(datasets,function(new_data) { # overwrite the original data with each subset and store all outputs in a list
    x$data <- new_data
    x})
}


risklabel <- expression(r[Omega](hat(theta)("·")))

# TODO add this throughout so that we can use png if needed
ggsv <- function(filename, plot, ...) {
  for (device in c("pdf", "png")) {
    # for (device in c("pdf")) {
    ggsave(plot, file = paste0(filename, ".", device), device = device, ...)
  }
}

