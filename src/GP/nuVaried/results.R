source("src/plotting.R")

model <- "spatial/GaussianProcess/nuVaried"
intermediates_path <- paste0("intermediates/", model, "/")
estimates_path     <- paste0(intermediates_path, "Estimates/")

parameter_labels = c(
  "τ"  = expression(hat(tau)), 
  "ρ"  = expression(hat(rho)),
  "ν"  = expression(hat(nu))
)

p <- length(parameter_labels)

img_path  <- paste0("img/", model)
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

missingness <- c("MCAR-fixedpattern", "MCAR-fixedproportion", "MNAR")

estimator_labels <- c(
  "analytic" = "MAP",
  "neuralEM" = "Neural EM",
  "neuralEncoding_fixedpattern" = "Encoding 1",
  "neuralEncoding_fixedproportion" = "Encoding 2",
  "neuralEncoding_variableproportion" = "Encoding 3"
)


# ---- Neural MAP vs. MAP obtained by maximising the analytic posterior density ----

df  <- estimates_path %>% paste0("estimates_complete.csv") %>% read.csv

marginal <- plotmarginals(df = df, 
                          estimator_labels = c("neural_MAP" = "neural MAP", "analytic_MAP" = "MAP"), 
                          parameter_labels = parameter_labels)
marginal <- ggarrange(plotlist = marginal, nrow = 1)
ggsave(
  marginal, file = "neuralEM_vs_analytic_complete.pdf",
  width = 8.5, height = 3.8, device = "pdf", path = img_path
)

# ---- Missing data: Entire parameter space ----

risk <- data.frame()
for (set in missingness) {

  df  <- estimates_path %>% paste0("estimates_", set, "_test.csv") %>% read.csv
  df  <- df %>% filter(estimator %in% names(estimator_labels))

  risk2 <- df %>% 
    mutate(loss = loss(estimate, truth)) %>% 
    group_by(estimator) %>% 
    summarise(risk = mean(loss), sd = sd(loss)/sqrt(length(loss)))
  risk2$missingness <- set
  risk <- rbind(risk, risk2)

  df       <- filter(df, estimator %in% c("neuralEM", "analytic"))
  marginal <- plotmarginals(df = df, estimator_labels = estimator_labels, parameter_labels = parameter_labels)
  marginal <- ggarrange(plotlist = marginal, nrow = 1)
  ggsave(
    marginal, file = paste0("neuralEM_vs_analytic_", set, ".pdf"),
    width = 8.5, height = 3.8, device = "pdf", path = img_path
  )
}
risk[, c("risk", "sd")] <- round(risk[, c("risk", "sd")], 2)
risk <- pivot_wider(risk, id_cols = "missingness", names_from = "estimator", values_from = c("risk", "sd"))
risk <- risk[match(missingness, risk$missingness), ] # order the rows by the missingngess 
write.csv(risk, row.names = F, file = paste0(img_path, "/risk.csv"))

# ---- Missing data: sampling distributions ----

loadestimates <- function(type) {
  df <- estimates_path %>% paste0("estimates_", type, "_scenarios.csv") %>% read.csv
  df$missingness <- type
  df
}

loaddata <- function(type) {
  df <- estimates_path %>% paste0("Z_", type, ".csv") %>% read.csv
  df$missingness <- type
  df
}

df <- loadestimates("MCAR-fixedpattern") %>%
  rbind(loadestimates("MCAR-fixedproportion")) %>%
  rbind(loadestimates("MNAR")) %>% 
  filter(estimator %in% names(estimator_labels))

zdf <- loaddata("MCAR-fixedpattern") %>%
  rbind(loaddata("MCAR-fixedproportion")) %>%
  rbind(loaddata("MNAR"))

N <- 16 # NB not ideal that this is hard coded; should probably save N, or x and y, in Julia
zdf$x <- rep(1:N, each = N)
zdf$y <- 1:N 


# TODO should fix the axes across rows. Also do this for the nuFixed case. Might
# be easiest to do this by foregoing plotdistribution() and calling facet_grid 
# with all of df. Think about whether this is really worth it later..
figures <- lapply(unique(df$k), function(kk) {
  
  df  <- df  %>% filter(k == kk)
  zdf <- zdf %>% filter(k == kk)
  
  data_MCAR0  <- field_plot(filter(zdf, j == 2, missingness == "MCAR-fixedpattern"))
  data_MCAR   <- field_plot(filter(zdf, j == 2, missingness == "MCAR-fixedproportion")) + labs(fill = "Z") + theme(legend.title.align=0.25, legend.title = element_text(face = "bold"))
  data_MNAR   <- field_plot(filter(zdf, j == 2, missingness == "MNAR"))
  data_legend <- get_legend(data_MCAR)
  data <- list(data_MCAR0, data_MCAR, data_MNAR)
  data <- lapply(data, function(gg) gg + theme(legend.position = "none"))
  
  box_MCAR0  <- plotdistribution(filter(df, missingness == "MCAR-fixedpattern"), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1) 
  box_MCAR   <- plotdistribution(filter(df, missingness == "MCAR-fixedproportion"), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1)
  box_MNAR   <- plotdistribution(filter(df, missingness == "MNAR"), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1) 
  box_legend <- get_legend(box_MCAR)
  box <- list(box_MCAR0, box_MCAR, box_MNAR)
  box <- lapply(box, function(gg) {
    gg$facet$params$nrow <- p
    gg$facet$params$strip.position <- "bottom"
    gg + 
      theme(legend.position = "none") +
      theme(
        strip.background = element_blank(),
        axis.title.y = element_blank()
      )
  })
  
  plotlist <- c(data, box)
  figure1  <- ggarrange(plotlist = plotlist, nrow = 2, ncol = 3, heights = c(1, p))
  figure2  <- ggarrange(data_legend, box_legend, ncol = 1, heights = c(1, 4))
  figure   <- ggarrange(figure1, figure2, widths = c(1, 0.2))
  figure
  
  ggsave(
    figure, file = paste0("missing_boxplots_k", kk, ".pdf"),
    width = 7.3, height = 7, device = "pdf", path = img_path
  )
  
  figure
})

