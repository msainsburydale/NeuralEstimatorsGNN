suppressMessages({
  library("NeuralEstimators")
  library("JuliaConnectoR")
  library("reshape2")
  library("ggplot2")
  library("dggrids")
  library("dplyr")
  library("FRK")
  library("sp")
  library("ggpubr")
  library("spdep") # poly2nb()
  library("spatstat.geom") # nndist()
  options(dplyr.summarise.inform = FALSE)
})

source("src/plotting.R")

# ---- Load data and estimates ----

clustered_data <- readRDS("~/NeuralEstimatorsGNN/intermediates/application/clustered_data.rds")
cells          <- readRDS("~/NeuralEstimatorsGNN/intermediates/application/cells.rds")

ML_estimates   <- read_csv("intermediates/application/ML_estimates.csv")
GNN_estimates  <- read_csv("intermediates/application/GNN_estimates.csv")
diff_estimates <- GNN_estimates - ML_estimates

process_estimates <- function(estimates) {
  estimates <- t(estimates)
  colnames(estimates) <- names(clustered_data)
  estimates <- melt(estimates, varnames = c("parameter", "id"), value.name = "estimate")
  return(estimates)
}

ML_estimates <- process_estimates(ML_estimates)
GNN_estimates <- process_estimates(GNN_estimates)
diff_estimates <- process_estimates(diff_estimates)


# ---- Plotting: point estimates ----

# Plot the point estimates
plot_estimates <- function(cells, estimates, param, limits = c(NA, NA)) {

  cells <- merge(cells, filter(estimates, parameter == param))

  suppressMessages({

    gg <- plot_spatial_or_ST(cells, column_names = "estimate", plot_over_world = T)[[1]]
    gg <- draw_world_custom(gg)
    gg <- gg +
      scale_fill_distiller(palette = "YlOrRd", na.value = NA, direction = 1, limits = limits) +
      theme(axis.title = element_blank(),
            panel.border = element_blank(),
            panel.background = element_blank(),
            legend.position = "top",
            legend.key.width = unit(1, 'cm'))

  })

  gg
}

rho_plot   <- plot_estimates(cells, ML_estimates, "ρ") + labs(fill = expression(hat(rho)))
sigma_plot <- plot_estimates(cells, ML_estimates, "σ") + labs(fill = expression(hat(sigma)))
tau_plot   <- plot_estimates(cells, ML_estimates, "τ") + labs(fill = expression(hat(sigma)[epsilon]))
ML_plot <- ggarrange(rho_plot, sigma_plot, tau_plot, nrow = 1)

rho_plot   <- plot_estimates(cells, GNN_estimates, "ρ") + labs(fill = expression(hat(rho)))
sigma_plot <- plot_estimates(cells, GNN_estimates, "σ") + labs(fill = expression(hat(sigma)))
tau_plot   <- plot_estimates(cells, GNN_estimates, "τ") + labs(fill = expression(hat(sigma)[epsilon]))
GNN_plot <- ggarrange(rho_plot, sigma_plot, tau_plot, nrow = 1)

ggarrange(ML_plot, GNN_plot)


# ---- Plotting: credible intervals ----

# Plot the credible interval widths
plot_ciwidth <- function(cells, estimates, param) {

  cells <- merge(cells, filter(estimates, parameter == param))

  gg <- plot_spatial_or_ST(cells, column_names = "estimate", plot_over_world = T)[[1]]
  gg <- draw_world_custom(gg)
  gg <- gg +
    # scale_fill_distiller(palette = "BrBG", na.value = NA, direction = -1) + # white centre value conflicts with missing colour (also white)
    scale_fill_distiller(palette = "GnBu", na.value = NA, direction = -1) +
    theme(axis.title = element_blank(),
          panel.border = element_blank(),
          panel.background = element_blank(),
          legend.position = "top",
          legend.key.width = unit(1, 'cm'))
  gg
}

rhoci_plot   <- plot_ciwidth(cells, estimates, "rho_ciwidth") + labs(fill = expression(atop(hat(rho) ~ "credible-", "interval width")))
sigmaci_plot <- plot_ciwidth(cells, estimates, "sigma_ciwidth") + labs(fill = expression(atop(hat(sigma) ~ "credible-", "interval width")))
tauci_plot   <- plot_ciwidth(cells, mutate(estimates, estimate = pmin(estimate, 0.5)), "tau_ciwidth") + labs(fill = expression(atop(hat(tau) ~ "credible-", "interval width")))

ggsv(
  ggpubr::ggarrange(
    rho_plot, sigma_plot, tau_plot,
    rhoci_plot, sigmaci_plot, tauci_plot,
    align = "hv", nrow = 2, ncol = p),
  filename = "estimates", width = 14, height = 6,
  path = img_path
)


# Plot the lower and upper quantiles
limits <- estimates %>% filter(parameter %in% c("rho_lower", "rho_upper")) %>% summarise(range(estimate))
limits <- limits[[1]]
rho_lower <- plot_estimates(cells, estimates, "rho_lower", limits) + labs(title = expression(hat(rho) *": lower bound" ), fill = "")
rho_upper <- plot_estimates(cells, estimates, "rho_upper", limits) + labs(title = expression(hat(rho) *": upper bound"), fill = "")
rho_ci <- ggpubr::ggarrange(rho_lower, rho_upper, align = "hv", nrow = 1, legend = "right", common.legend = T)

limits <- estimates %>% filter(parameter %in% c("sigma_lower", "sigma_upper")) %>% summarise(range(estimate))
limits <- limits[[1]]
sigma_lower <- plot_estimates(cells, estimates, "sigma_lower", limits) + labs(title = expression(hat(sigma) *": lower bound" ), fill = "")
sigma_upper <- plot_estimates(cells, estimates, "sigma_upper", limits) + labs(title = expression(hat(sigma) *": upper bound"), fill = "")
sigma_ci <- ggpubr::ggarrange(sigma_lower, sigma_upper, align = "hv", nrow = 1, legend = "right", common.legend = T)

estimates_tau <- estimates %>%
  filter(parameter %in% c("tau_lower", "tau_upper")) %>%
  mutate(estimate = pmin(estimate, 0.6))
limits <- estimates_tau %>% summarise(range(estimate))
limits <- limits[[1]]
tau_lower <- plot_estimates(cells, estimates_tau, "tau_lower", limits) + labs(title = expression(hat(tau) *": lower bound"), fill = "")
tau_upper <- plot_estimates(cells, estimates_tau, "tau_upper", limits) + labs(title = expression(hat(tau) *": upper bound"), fill = "")
tau_ci <- ggpubr::ggarrange(tau_lower, tau_upper, align = "hv", nrow = 1, legend = "right", common.legend = T)

figure <- ggpubr::ggarrange(rho_ci, sigma_ci, tau_ci, ncol = 1)

ggsv(figure, filename = "intervals", width = 8, height = 7, path = img_path)