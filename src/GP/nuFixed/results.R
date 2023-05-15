source("src/plotting.R")

model <- "spatial/GaussianProcess/nuFixed"
intermediates_path <- paste0("intermediates/", model, "/")
estimates_path     <- paste0(intermediates_path, "Estimates/")
img_path  <- paste0("img/", model)
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

parameter_labels = c(
  "τ"  = expression(hat(tau)), 
  "ρ"  = expression(hat(rho))
)

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

# ---- Missing data: Joint distributions ----

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

figures <- lapply(unique(df$k), function(K) {
  
  df  <- df  %>% filter(k == K)
  zdf <- zdf %>% filter(k == K)
  
  data_MCAR0  <- field_plot(filter(zdf, j == 1, missingness == "MCAR-fixedpattern")) #+ labs(title = "MCAR\nFixed pattern")
  data_MCAR   <- field_plot(filter(zdf, j == 1, missingness == "MCAR-fixedproportion")) #+ labs(title = "MCAR\nVariable pattern")
  data_MNAR   <- field_plot(filter(zdf, j == 1, missingness == "MNAR")) #+ labs(title = "\nMissing block")
  data_MCAR   <- data_MCAR + labs(fill = "Z") + theme(legend.title.align=0.25, legend.title = element_text(face = "bold"))
  data_legend <- get_legend(data_MCAR)
  data <- list(data_MCAR0, data_MCAR, data_MNAR)
  data <- lapply(data, function(gg) gg + 
                   theme(legend.position = "none") + 
                   theme(plot.title = element_text(hjust = 0.5)))
  
  joint_MCAR0  <- plotdistribution(filter(df, missingness == "MCAR-fixedpattern"), type = "scatter", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1)[[1]] # + scale_estimator(df)
  joint_MCAR   <- plotdistribution(filter(df, missingness == "MCAR-fixedproportion"), type = "scatter", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1)[[1]] # + scale_estimator(df)
  joint_MNAR   <- plotdistribution(filter(df, missingness == "MNAR"), type = "scatter", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1)[[1]] # + scale_estimator(df)
  joint_legend <- get_legend(joint_MCAR)
  joint <- list(joint_MCAR0, joint_MCAR, joint_MNAR)
  joint <- lapply(joint, function(gg) gg + theme(legend.position = "none"))
  
  plotlist <- c(data, joint)
  figure1  <- ggarrange(plotlist = plotlist, nrow = 2, ncol = 3)
  figure2  <- ggarrange(data_legend, joint_legend, ncol = 1)
  figure   <- ggarrange(figure1, figure2, widths = c(1, 0.2))
  
  ggsave(
    figure, file = paste0("missing_joint_k", K, ".pdf"),
    width = 14, height = 6, device = "pdf", path = img_path
  )
  
  box_MCAR0  <- plotdistribution(filter(df, missingness == "MCAR-fixedpattern"), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1) # + scale_estimator(df)
  box_MCAR   <- plotdistribution(filter(df, missingness == "MCAR-fixedproportion"), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1) # + scale_estimator(df)
  box_MNAR   <- plotdistribution(filter(df, missingness == "MNAR"), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1) # + scale_estimator(df)
  box_legend <- get_legend(box_MCAR)
  box <- list(box_MCAR0, box_MCAR, box_MNAR)
  box <- lapply(box, function(gg) {
    gg$facet$params$nrow <- 2
    gg$facet$params$strip.position <- "bottom"
    gg + 
      theme(legend.position = "none") +
      theme(
        strip.background = element_blank(),
        # strip.text.x = element_blank(),
        axis.title.y = element_blank()
      )
  })
  
  plotlist <- c(data, box)
  figure1  <- ggarrange(plotlist = plotlist, nrow = 2, ncol = 3, heights = c(1, 2))
  figure2  <- ggarrange(data_legend, box_legend, ncol = 1, heights = c(1, 2.5))
  figure   <- ggarrange(figure1, figure2, widths = c(1, 0.2))
  figure
  
  ggsave(
    figure, file = paste0("missing_boxplots_k", K, ".pdf"),
    width = 8.5, height = 6, device = "pdf", path = img_path
  )
  
  # ---- joint and marginals merged into one ----
  
  joint <- lapply(joint, function(gg) ggMarginal(gg, groupFill = TRUE, groupColour = TRUE, type = "density", alpha = 0.5, position = "identity"))
  joint <- lapply(joint, function(gg) as.ggplot(gg))
  
  plotlist <- c(data, joint)
  figure1  <- ggarrange(plotlist = plotlist, nrow = 2, ncol = 3)
  figure2  <- ggarrange(data_legend, joint_legend, ncol = 1)
  figure   <- ggarrange(figure1, figure2, widths = c(1, 0.25))
  
  ggsave(
    figure, file = paste0("missing_jointmarginals_k", K, ".pdf"),
    width = 11, height = 6, device = "pdf", path = img_path
  )
  
  figure
})




