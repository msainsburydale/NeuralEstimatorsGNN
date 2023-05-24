library("optparse")
option_list <- list(
  make_option("--model", type="character", default=NULL, metavar="character")
)
opt_parser  <- OptionParser(option_list=option_list)
model       <- parse_args(opt_parser)$model

int_path <- paste0("intermediates/", model)
img_path <- paste0("img/", model)
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

loadestimates <- function(set, type = "scenarios") {
  df <- read.csv(paste0(int_path, "/estimates_", type, "_", set, ".csv"))
  df$set <- set
  df
}

source("src/plotting.R")

boldtheta <- bquote(bold(theta))
estimator_labels <- c(
  "GNN" = "GNN",
  "CNN" = "CNN",
  "MAP" = "MAP"
)
estimators <- names(estimator_labels)

# ---- Risk function ----

df <- loadestimates("gridded", "test") %>%
  rbind(loadestimates("uniform", "test")) %>%
  rbind(loadestimates("quadrants", "test")) %>% 
  rbind(loadestimates("mixedsparsity", "test")) %>% 
  rbind(loadestimates("cup", "test")) %>% 
  filter(estimator %in% estimators)

## Bayes risk with respect to absolute error
df %>%
  mutate(loss = abs(estimate - truth)) %>% 
  group_by(set, estimator) %>% 
  summarise(risk = mean(loss), sd = sd(loss)/sqrt(length(loss))) %>%
  write.csv(file = paste0(img_path, "/risk.csv"), row.names = F)


## RMSE  ## TODO can I get an estimate of the sd? 
df %>%
  mutate(loss = (estimate - truth)^2) %>% 
  group_by(set, estimator) %>% 
  summarise(RMSE = sqrt(mean(loss))) %>%
  write.csv(file = paste0(img_path, "/RMSE.csv"), row.names = F)


# ---- Sampling distributions ----

loaddata <- function(set) {
  df <- read.csv(paste0(int_path, "/Z_", set, ".csv"))
  df$set <- set
  df
}

df <- loadestimates("gridded") %>%
  rbind(loadestimates("uniform")) %>%
  rbind(loadestimates("quadrants")) %>% 
  rbind(loadestimates("mixedsparsity")) %>% 
  rbind(loadestimates("cup")) %>% 
  filter(estimator %in% estimators)

zdf <- loaddata("gridded") %>%
  rbind(loaddata("uniform")) %>%
  rbind(loaddata("quadrants")) %>%
  rbind(loaddata("mixedsparsity")) %>%
  rbind(loaddata("cup"))

figures <- lapply(unique(df$k), function(K) {
  
  df  <- df  %>% filter(k == K)
  zdf <- zdf %>% filter(k == K)
  
  ggz_1  <- field_plot(filter(zdf, set == "gridded"), regular = T) # NB set regular = F for consistency
  ggz_2  <- field_plot(filter(zdf, set == "uniform"), regular = F) 
  ggz_3  <- field_plot(filter(zdf, set == "quadrants"), regular = F)
  ggz_4  <- field_plot(filter(zdf, set == "mixedsparsity"), regular = F)
  ggz_5  <- field_plot(filter(zdf, set == "cup"), regular = F)
  ggz_1  <- ggz_1 + labs(fill = "Z") + theme(legend.title.align = 0.25, legend.title = element_text(face = "bold"))
  data_legend <- get_legend(ggz_1)
  data <- list(ggz_1, ggz_2, ggz_3, ggz_4, ggz_5)
  data <- lapply(data, function(gg) gg + 
                   theme(legend.position = "none") + 
                   theme(plot.title = element_text(hjust = 0.5)) + 
                   coord_fixed())
  
  
  
  # Marginal distributions
  box_1  <- plotdistribution(filter(df, set == "gridded"), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1) + scale_estimator(df)
  box_2  <- plotdistribution(filter(df, set == "uniform"), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1)  + scale_estimator(df)
  box_3  <- plotdistribution(filter(df, set == "quadrants"), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1)  + scale_estimator(df)
  box_4  <- plotdistribution(filter(df, set == "mixedsparsity"), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1) + scale_estimator(df)
  box_5  <- plotdistribution(filter(df, set == "cup"), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1) + scale_estimator(df)
  box_legend <- get_legend(box_1)
  box <- list(box_1, box_2, box_3, box_4, box_5)
  box <- lapply(box, function(gg) {
    gg$facet$params$nrow <- 2
    gg$facet$params$strip.position <- "bottom"
    gg + 
      theme(legend.position = "none") +
      theme(
        strip.background = element_blank(),
        axis.title.y = element_blank()
      )
  })
  
  plotlist <- c(data, box)
  figure1  <- ggarrange(plotlist = plotlist, nrow = 2, ncol = 5, heights = c(1.25, 2))
  figure2  <- ggarrange(data_legend, box_legend, ncol = 1, heights = c(1, 2.5))
  figure   <- ggarrange(figure1, figure2, widths = c(1, 0.15))
  figure
  
  ggsave(
    figure, 
    file = paste0("samplingdistributions_marginal", K, ".pdf"),
    width = 12.5, height = 6, device = "pdf", path = img_path
  )
  
  # Joint distributions
  joint_1  <- plotdistribution(filter(df, set == "gridded"), type = "scatter", parameter_labels = parameter_labels, estimator_labels = estimator_labels)[[1]] + scale_estimator(df)
  joint_2  <- plotdistribution(filter(df, set == "uniform"), type = "scatter", parameter_labels = parameter_labels, estimator_labels = estimator_labels)[[1]] + scale_estimator(df)
  joint_3  <- plotdistribution(filter(df, set == "quadrants"), type = "scatter", parameter_labels = parameter_labels, estimator_labels = estimator_labels)[[1]] + scale_estimator(df)
  joint_4  <- plotdistribution(filter(df, set == "mixedsparsity"), type = "scatter", parameter_labels = parameter_labels, estimator_labels = estimator_labels)[[1]] + scale_estimator(df)
  joint_5  <- plotdistribution(filter(df, set == "cup"), type = "scatter", parameter_labels = parameter_labels, estimator_labels = estimator_labels)[[1]] + scale_estimator(df)
  joint_legend <- get_legend(joint_1)
  joint <- list(joint_1, joint_2, joint_3, joint_4, joint_5)
  joint <- lapply(joint, function(gg) {
    gg$facet$params$nrow <- 2
    gg$facet$params$strip.position <- "bottom"
    gg + 
      theme(legend.position = "none") +
      theme(
        strip.background = element_blank()
      )
  })
  
  plotlist <- c(data, joint)
  figure1  <- ggarrange(plotlist = plotlist, nrow = 2, ncol = 5, align = "hv")
  figure2  <- ggarrange(data_legend, joint_legend, ncol = 1, heights = c(1, 1))
  figure   <- ggarrange(figure1, figure2, widths = c(1, 0.15))
  figure
  
  ggsave(
    figure, 
    file = paste0("samplingdistributions_joint", K, ".pdf"),
    width = 12.5, height = 4, device = "pdf", path = img_path
  )
  
  figure
})




