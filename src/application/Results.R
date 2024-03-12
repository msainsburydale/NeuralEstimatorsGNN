suppressMessages({
  library("NeuralEstimators")
  library("JuliaConnectoR")
  library("reshape2")
  library("egg")
  library("ggpubr")
  library("ggplot2")
  library("dggrids")
  library("dplyr")
  library("FRK")
  library("sp")
  library("spdep") # poly2nb()
  library("spatstat.geom") # nndist()
  options(dplyr.summarise.inform = FALSE)
})

source("src/plotting.R")

img_path <- "img/application"
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

# ---- Load data and estimates ----

clustered_data <- readRDS("intermediates/application/clustered_data.rds")
cells          <- readRDS("intermediates/application/cells.rds")

ML_estimates   <- read.csv("intermediates/application/ML_estimates.csv")
GNN_estimates  <- read.csv("intermediates/application/GNN_estimates.csv")

rho_limit   <- range(c(ML_estimates$ρ, GNN_estimates$ρ))
sigma_limit <- range(c(ML_estimates$σ, GNN_estimates$σ))
tau_limit <- range(c(ML_estimates$τ, GNN_estimates$τ))

GNN_estimates$rho_ciwidth <- GNN_estimates$ρ_upper - GNN_estimates$ρ_lower
GNN_estimates$sigma_ciwidth <- GNN_estimates$σ_upper - GNN_estimates$σ_lower
GNN_estimates$tau_ciwidth <- GNN_estimates$τ_upper - GNN_estimates$τ_lower

GNN_estimates$ρ_error <- GNN_estimates$ρ - ML_estimates$ρ
GNN_estimates$σ_error <- GNN_estimates$σ - ML_estimates$σ
GNN_estimates$τ_error <- GNN_estimates$τ - ML_estimates$τ

process_estimates <- function(estimates) {
  estimates <- t(estimates)
  colnames(estimates) <- names(clustered_data)
  estimates <- melt(estimates, varnames = c("parameter", "id"), value.name = "estimate")
  return(estimates)
}

ML_estimates <- process_estimates(ML_estimates)
GNN_estimates <- process_estimates(GNN_estimates)


# ---- Plotting: point estimates ----

# Plot the point estimates
plot_estimates <- function(cells, estimates, param, limits = c(NA, NA)) {

  estimates$estimate <- pmin(estimates$estimate, limits[2], na.rm = TRUE)
  cells <- merge(cells, filter(estimates, parameter == param))

  suppressMessages({

    gg <- plot_spatial_or_ST(cells, column_names = "estimate", plot_over_world = T)[[1]]
    gg <- draw_world_custom(gg)
    gg <- gg +
      theme(axis.title = element_blank(),
            panel.border = element_blank(),
            panel.background = element_blank(),
            legend.position = "top",
            legend.key.width = unit(1, 'cm'))

    if (grep("error", param)) {
      gg <- gg + scale_fill_gradient2(low = 'blue', mid = 'white', high = 'red', midpoint = 0, limits = c(-limits[2], limits[2]), na.value = NA)
    } else {
      gg <- gg + scale_fill_distiller(palette = "YlOrRd", na.value = NA, direction = 1, limits = limits)
    }

  })

  gg
}

rho_plot1   <- plot_estimates(cells, GNN_estimates, "ρ", rho_limit) + labs(fill = expression(hat(rho)))
sigma_plot1 <- plot_estimates(cells, GNN_estimates, "σ", sigma_limit) + labs(fill = expression(hat(sigma)))
tau_plot1   <- plot_estimates(cells, GNN_estimates, "τ", tau_limit) + labs(fill = expression(hat(sigma)[epsilon]))

rho_plot2   <- plot_estimates(cells, ML_estimates, "ρ", rho_limit) + theme(legend.position = "none")
sigma_plot2 <- plot_estimates(cells, ML_estimates, "σ", sigma_limit) + theme(legend.position = "none")
tau_plot2   <- plot_estimates(cells, ML_estimates, "τ", tau_limit) + theme(legend.position = "none")

fig <- egg::ggarrange(rho_plot1, sigma_plot1, tau_plot1,
                      rho_plot2, sigma_plot2, tau_plot2,
                      nrow = 2)
ggsv(fig, filename = "GNN_ML", width = 10, height = 4, path = img_path)

## Error maps
rho_plot3   <- plot_estimates(cells, GNN_estimates, "ρ_error", rho_limit) + labs(fill = expression(hat(rho)[GNN] - hat(rho)[ML]))
sigma_plot3 <- plot_estimates(cells, GNN_estimates, "σ_error", sigma_limit) + labs(fill = expression(hat(sigma)[GNN] - hat(sigma)[ML]))
tau_plot3   <- plot_estimates(cells, GNN_estimates, "τ_error", tau_limit) + labs(fill = expression(hat(sigma)[epsilon]))
fig <- egg::ggarrange(rho_plot3, sigma_plot3, tau_plot3, nrow = 1)

# ---- Plotting: credible intervals ----

# Plot the credible interval widths
plot_ciwidth <- function(cells, estimates, param) {

  cells <- merge(cells, filter(estimates, parameter == param))

  gg <- plot_spatial_or_ST(cells, column_names = "estimate", plot_over_world = T)[[1]]
  gg <- draw_world_custom(gg)
  suppressMessages({
  gg <- gg +
    # scale_fill_distiller(palette = "BrBG", na.value = NA, direction = -1) + # white centre value conflicts with missing colour (also white)
    scale_fill_distiller(palette = "GnBu", na.value = NA, direction = -1) +
    theme(axis.title = element_blank(),
          panel.border = element_blank(),
          panel.background = element_blank(),
          legend.position = "top",
          legend.key.width = unit(1, 'cm'))
  })
  gg
}


rhoci_plot   <- plot_ciwidth(cells, GNN_estimates, "rho_ciwidth") + labs(fill = expression(atop(hat(rho) ~ "credible-", "interval width")))
sigmaci_plot <- plot_ciwidth(cells, GNN_estimates, "sigma_ciwidth") + labs(fill = expression(atop(hat(sigma) ~ "credible-", "interval width")))
tauci_plot   <- plot_ciwidth(cells, mutate(GNN_estimates, estimate = pmin(estimate, 0.5)), "tau_ciwidth") + labs(fill = expression(atop(hat(sigma)[epsilon] ~ "credible-", "interval width")))

fig <-  egg::ggarrange(rho_plot1, sigma_plot1, tau_plot1,
                       rhoci_plot, sigmaci_plot, tauci_plot,
                       nrow = 2)

ggsv(fig, filename = "estimates", width = 12, height = 6, path = img_path)


# Plot the lower and upper quantiles

#TODO align the legends

limits <- GNN_estimates %>% filter(parameter %in% c("ρ_lower", "ρ_upper")) %>% summarise(range(estimate))
limits <- limits[[1]]
rho_lower <- plot_estimates(cells, GNN_estimates, "ρ_lower", limits) + labs(title = expression(hat(rho) *": lower bound" ), fill = "")
rho_upper <- plot_estimates(cells, GNN_estimates, "ρ_upper", limits) + labs(title = expression(hat(rho) *": upper bound"), fill = "")
rho_ci <- ggpubr::ggarrange(rho_lower, rho_upper, align = "hv", nrow = 1, legend = "right", common.legend = T)

limits <- GNN_estimates %>% filter(parameter %in% c("σ_lower", "σ_upper")) %>% summarise(range(estimate))
limits <- limits[[1]]
sigma_lower <- plot_estimates(cells, GNN_estimates, "σ_lower", limits) + labs(title = expression(hat(sigma) *": lower bound" ), fill = "")
sigma_upper <- plot_estimates(cells, GNN_estimates, "σ_upper", limits) + labs(title = expression(hat(sigma) *": upper bound"), fill = "")
sigma_ci <- ggpubr::ggarrange(sigma_lower, sigma_upper, align = "hv", nrow = 1, legend = "right", common.legend = T)

estimates_tau <- GNN_estimates %>%
  filter(parameter %in% c("τ_lower", "τ_upper")) %>%
  mutate(estimate = pmin(estimate, 0.6)) # TODO make this consistent with above
limits <- estimates_tau %>% summarise(range(estimate))
limits <- limits[[1]]
tau_lower <- plot_estimates(cells, estimates_tau, "τ_lower", limits) + labs(title = expression(hat(sigma)[epsilon] *": lower bound"), fill = "")
tau_upper <- plot_estimates(cells, estimates_tau, "τ_upper", limits) + labs(title = expression(hat(sigma)[epsilon] *": upper bound"), fill = "")
tau_ci <- ggpubr::ggarrange(tau_lower, tau_upper, align = "hv", nrow = 1, legend = "right", common.legend = T)

fig <- ggpubr::ggarrange(rho_ci, sigma_ci, tau_ci, ncol = 1)

ggsv(fig, filename = "intervals", width = 8, height = 7, path = img_path)
