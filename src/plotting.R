library("NeuralEstimators")
library("ggplot2")
library("dplyr")
library("ggpubr")
library("viridis")
library("tidyr")

# The simulations may be highly varied in magnitude, so we need to
# use an independent colour scale. This means that we can't use facet_wrap().
field_plot <- function(field, regular = TRUE, n = nrow(field)) { #TODO specify the number of missing locations; these will be coloured white
  
  N <- nrow(field)
  if (n > N) stop("The number of observed locations, n, must be less than or equal to the total number of locations, N = nrow(field)")
  idx <- sample(N, n, replace = FALSE)
  field <- field[idx, ]
  
  gg <- ggplot(field, aes(x = x, y = y))
  
  if (regular) {
    gg <- gg +
      geom_tile(aes(fill = Z)) +
      scale_fill_viridis_c(option = "magma")
  } else {
    gg <- gg +
      geom_point(aes(colour = Z)) +
      scale_colour_viridis_c(option = "magma")
  }
  
  gg <- gg +
    labs(fill = "", x = " ", y = " ") +
    theme_bw() +
    theme(
      panel.grid = element_blank(),
      panel.border = element_blank(),
      axis.ticks   =  element_blank(),
      axis.text    = element_blank(),
      legend.position = "right",
      strip.background = element_blank(),
      strip.text.x = element_blank()
    ) +
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
