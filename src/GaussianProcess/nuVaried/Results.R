library("NeuralEstimators")
library("ggplot2")
library("dplyr")

model <- "GaussianProcess/nuVaried"
intermediates_path <- paste0("intermediates/", model, "/")
estimates_path     <- paste0(intermediates_path, "Estimates/")
img_path           <- paste("img", model, sep = "/")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

parameter_labels = c(
  "σ"  = expression(sigma[epsilon]),
  "ρ"  = expression(rho),
  "ν"  = expression(nu)
)

# ---- Common functions ----

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


# ---- Risk function ----

df <- estimates_path %>% paste0("estimates_test_complete.csv") %>% read.csv
missing_path <- paste0(estimates_path, "estimates_test_missing.csv")
if (file.exists(missing_path)) df <- rbind(df, read.csv(missing_path))

all_loss <- c("MAE", "RMSE", "MSE", "MAD", "zeroone")

risk_plots <- lapply(all_loss, function(loss) {

  gg <- df %>%
    plotrisk(parameter_labels = parameter_labels, loss = eval(parse(text = loss))) +
    theme(
      strip.text = element_text(size = 17),
      axis.title = element_text(size = 17),
      axis.text = element_text(size = 15)
    )

  ggsave(
    gg, width = 12, height = 4, device = "pdf", path = img_path,
    file = paste0(loss, "_vs_m.pdf"),
  )
})


# ---- Marginal distributions: boxplots ----

df <- estimates_path %>% paste0("estimates_scenarios_complete.csv") %>% read.csv
missing_path    <- paste0(estimates_path, "estimates_scenarios_missing.csv")
likelihood_path <- paste0(estimates_path, "estimates_scenarios_missing_MAP.csv")
if (file.exists(missing_path))    df <- rbind(df, read.csv(missing_path))
if (file.exists(likelihood_path)) df <- rbind(df, read.csv(likelihood_path))

lapply(unique(df$k), function(K) {

  ggsave(
    plotdistribution(df %>% filter(k == K, m == 150), parameter_labels = parameter_labels),
    width = 8, height = 4, device = "pdf", path = img_path,
    file = paste0("boxplot", K, ".pdf")
  )
})

# df %>% filter(k == 1, m == 150) %>%
#   plotdistribution(parameter_labels = parameter_labels) %>%
#   splitfacet()

plotdistribution(df %>% filter(k == K, m == 150), parameter_labels = parameter_labels)


# ---- Joint distributions: scatterplots ----

n <- 200 # TODO should load this from Julia

# df <- estimates_path %>% paste0("estimates_scenarios_complete.csv") %>% read.csv
# missing_path <- paste0(estimates_path, "estimates_scenarios_missing.csv")
# if (file.exists(missing_path)) df <- rbind(df, read.csv(missing_path))

# filter the estimates to a subset sample sizes
all_m  <- 150
df <- df %>% filter(m %in% all_m)

# Load realisations from the model
fields_scenarios <- paste0(intermediates_path, "fields.csv") %>% read.csv

# Single scenario:
plotlist <- df %>% filter(m == 150, k == df$k[1]) %>% plotdistribution(type = "scatter", parameter_labels = parameter_labels)
plotlist <- lapply(plotlist, function(gg) gg + theme(legend.text.align = 0))
figure   <- ggarrange(plotlist = plotlist, nrow = 1, common.legend = TRUE, legend = "right")
ggsave(figure, file = "Scatterplot_m150_singleScenario.pdf",
       width = 9, height = 2.8, path = img_path, device = "pdf")

# Several scenarios and with a field realisation:
plotscenario <- function(df, field, type = "scatter", ...) {

  fieldplot <- field_plot(field, ...) + theme(legend.position = "top")
  distplots <- plotdistribution(df, type = type, parameter_labels = parameter_labels)

  if (type == "box") distplots <- splitfacet(distplots)

  legend.grob <<- get_legend(distplots[[1]]) # cheeky super-assignment

  plotlist <- c(list(fieldplot), distplots)

  return(plotlist)
}


snk <- lapply(c("box", "scatter"), function(type) {
  lapply(all_m, function(j) {

    plotlist <- lapply(unique(df$k), function(scen) {

      df     <- df %>% filter(k == scen, m == j)
      field  <- fields_scenarios %>% filter(scenario == scen, replicate == 1)

      ggarrange(plotlist = plotscenario(df, field, type = type, n = n), legend = "none", nrow = 1, align = "hv")
    })

    fig <- ggarrange(plotlist = plotlist, legend.grob = legend.grob, ncol = 1)

    ggsave(fig, file = paste0(type, "plot_m", j, ".pdf"),
           width = 8.3, height = 1.75 * length(unique(df$k)), path = img_path, device = "pdf")

    return(fig)
  })
})
