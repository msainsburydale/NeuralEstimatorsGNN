source("src/plotting.R")

model <- "GaussianProcess/nuFixed"
intermediates_path <- paste0("intermediates/", model, "/")
estimates_path     <- paste0(intermediates_path, "Estimates/")
img_path           <- paste("img", model, sep = "/")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

parameter_labels = c(
  "τ"  = expression(tau),
  "ρ"  = expression(rho)
)

estimator_labels <- c(
  "neural" = "neural",
  "MAP" = "analytic"
)


df <- paste0(estimates_path, "estimates_missing_neural.csv") %>% read.csv
dfmap <- paste0(estimates_path, "estimates_missing_MAP.csv") %>% read.csv
df <- rbind(df, dfmap)

# convert to wide format based on the parameters
df <- df %>%
  pivot_wider(names_from = parameter, values_from = c("estimate", "truth")) %>%
  as.data.frame

plotlist <- lapply(unique(df$k), function(K) {

  df        <- filter(df, k == K)

  # TODO force this to be square (but with diff)
  joint <- ggplot(df) +
    geom_point(aes(estimate_τ, estimate_ρ, colour = estimator), alpha = 0.5) +
    geom_point(aes(truth_τ, truth_ρ), col = "red", shape = "+", size = 8) +
    labs(colour = "",
         # x = parameter_labels[[1]], y = parameter_labels[[2]]
         x = as.expression(bquote(hat(.(parameter_labels[[1]])))),
         y = as.expression(bquote(hat(.(parameter_labels[[2]]))))
         ) +
    scale_colour_viridis(discrete = TRUE,  labels = estimator_labels) +
    theme_bw() +
    theme(aspect.ratio=1)

  # convert to a wider format based on the estimator names
  df <- df %>%
    pivot_wider(names_from = estimator, values_from = paste("estimate", names(parameter_labels), sep = "_")) %>%
    as.data.frame

  # joint difference
  parameters <- as.list(parameter_labels)
  estimators <- rev(names(estimator_labels))
  p <- length(parameter_labels)
  columns <- lapply(1:p, function(i) {
    paste("estimate", names(parameter_labels)[i], estimators, sep = "_")
  })
  param1 <- df[, columns[[1]][2]] - df[, columns[[1]][1]]
  param2 <- df[, columns[[2]][2]] - df[, columns[[2]][1]]
  tmp <- data.frame(param1, param2)
  joint_diff <- ggplot(tmp) +
    geom_point(aes_string(param1, param2)) +
    labs(
      x = as.expression(bquote(hat(.(parameters[[1]]))[.(estimator_labels[2])] ~ "-" ~ hat(.(parameters[[1]]))[.(estimator_labels[1])])),
      y = as.expression(bquote(hat(.(parameters[[2]]))[.(estimator_labels[2])] ~ "-" ~ hat(.(parameters[[2]]))[.(estimator_labels[1])])),
    ) +
    geom_vline(xintercept = 0) + geom_hline(yintercept = 0) +
    theme_bw() +
    theme(strip.background = element_blank(), strip.text.x = element_blank())

  marginal <- lapply(1:p, function(i) {

    columns <- paste("estimate", names(parameter_labels)[i], estimators, sep = "_")
    lmts <- range(df[, columns])

    ggplot(df) +
      geom_point(aes_string(columns[1], columns[2])) +
      geom_abline(colour = "red") +
      labs(
        x = as.expression(bquote(hat(.(parameters[[i]]))[.(estimator_labels[1])])),
        y = as.expression(bquote(hat(.(parameters[[i]]))[.(estimator_labels[2])]))
      ) +
      theme_bw() +
      theme(strip.background = element_blank(), strip.text.x = element_blank()) +
      coord_fixed(xlim = lmts, ylim = lmts)
  })

  plotlist <- c(marginal, list(joint_diff), list(joint))
  fig <- ggarrange(
    plotlist = plotlist, nrow = 1,
    widths = c(1, 1, 1, 1.5)
    )

  ggsave(
    fig, file = paste0("neural_vs_analytic_MAP_k", K, ".pdf"),
    device = "pdf", path = img_path,
    # width = 11.4, height = 3
    width = 15, height = 3
  )

  fig
})


ggsave(
  ggarrange(plotlist = plotlist, ncol = 1),
  file = "neural_vs_analytic_MAP_all.pdf",
  device = "pdf", path = img_path,
  # width = 11.4, height = 15
  width = 15, height = 15
)









# # ---- Neural estimator: complete data ----
#
# df <- estimates_path %>% paste0("estimates_scenarios_complete.csv") %>% read.csv
# M <- 150
#
# lapply(unique(df$k), function(K) {
#   ggsave(
#     plotdistribution(df %>% filter(k == K, m == M), parameter_labels = parameter_labels),
#     file = paste0("complete_boxplot", K, ".pdf"),
#     width = 8, height = 4, device = "pdf", path = img_path
#   )
# })
#
# # ---- Neural estimator: missing data ----
#
# df <- paste0(estimates_path, "estimates_scenarios_missing.csv") %>% read.csv
# # MAP_path <- paste0(estimates_path, "estimates_scenarios_missing_MAP.csv")
# # if (file.exists(MAP_path)) df <- rbind(df, read.csv(MAP_path))
#
#
# # Marginal distributions
# lapply(unique(df$k), function(K) {
#   ggsave(
#     plotdistribution(df %>% filter(k == K), parameter_labels = parameter_labels, type = "box"),
#     width = 8, height = 4, device = "pdf", path = img_path,
#     file = paste0("boxplot", K, ".pdf")
#   )
# })
#
# # df %>% filter(k == 1, m == 150) %>%
# #   plotdistribution(parameter_labels = parameter_labels) %>%
# #   splitfacet()
#
#
# # ---- Joint distributions: scatterplots ----
#
# n <- 200 # TODO should load this from Julia
#
# # Load realisations from the model
# fields_scenarios <- paste0(intermediates_path, "fields.csv") %>% read.csv
#
# # # Single scenario:
# # plotlist <- df %>% filter(k == df$k[1]) %>% plotdistribution(type = "scatter", parameter_labels = parameter_labels)
# # plotlist <- lapply(plotlist, function(gg) gg + theme(legend.text.align = 0))
# # figure   <- ggarrange(plotlist = plotlist, nrow = 1, common.legend = TRUE, legend = "right")
# # ggsave(figure, file = "Scatterplot_m150_singleScenario.pdf",
# #        width = 9, height = 2.8, path = img_path, device = "pdf")
#
# # Several scenarios and with a field realisation:
# plotscenario <- function(df, field, type = "scatter", ...) {
#
#   fieldplot <- field_plot(field, ...) + theme(legend.position = "top")
#   distplots <- plotdistribution(df, type = type, parameter_labels = parameter_labels)
#
#   if (type == "box") distplots <- splitfacet(distplots)
#
#   legend.grob <<- get_legend(distplots[[1]]) # cheeky super-assignment
#
#   plotlist <- c(list(fieldplot), distplots)
#
#   return(plotlist)
# }
#
# all_m <- 1
# snk <- lapply(c("box", "scatter"), function(type) {
#   lapply(all_m, function(j) {
#
#     plotlist <- lapply(unique(df$k), function(scen) {
#
#       df     <- df %>% filter(k == scen, m == j)
#       field  <- fields_scenarios %>% filter(scenario == scen, replicate == 1)
#
#       ggarrange(plotlist = plotscenario(df, field, type = type, n = n), legend = "none", nrow = 1, align = "hv")
#     })
#
#     fig <- ggarrange(plotlist = plotlist, legend.grob = legend.grob, ncol = 1)
#
#     ggsave(fig, file = paste0(type, "plot_m", j, ".pdf"),
#            width = 8.3, height = 1.75 * length(unique(df$k)), path = img_path, device = "pdf")
#
#     return(fig)
#   })
# })
