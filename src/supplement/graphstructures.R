model       <- "GP/nuFixed"
int_path <- paste("intermediates/supplement/graphstructures", model, sep = "/")
img_path <- paste("img/supplement/graphstructures", model, sep = "/")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)



loadestimates <- function(set, type = "scenarios") {
  df <- read.csv(paste0(int_path, "/estimates_", type, "_", set, ".csv"))
  df$set <- set
  df
  # Relabel to control the order of boxplots
  # df$estimator <- as.factor(df$estimator)
  # levels(df$estimator) <- c("GNN_S", "GNN_Svariable", "GNN_Sclustered", "MAP")
  df$estimator[df$estimator == "GNN_S"] <- "GNN_S1"
  df$estimator[df$estimator == "GNN_Svariable"] <- "GNN_S2"
  df$estimator[df$estimator == "GNN_Sclustered"] <- "GNN_S3"
  df$estimator[df$estimator == "GNN_Smatern"] <- "GNN_S4"

  df
}

source("src/plotting.R")

# ---- Risk function ----

df <- loadestimates("S", "test") %>%
  rbind(loadestimates("Stilde", "test")) %>%
  rbind(loadestimates("Sclustered", "test")) %>%
  filter(estimator %in% estimators)

## Bayes risk with respect to absolute error
df %>%
  mutate(loss = abs(estimate - truth)) %>% 
  group_by(set, estimator) %>%
  summarise(risk = mean(loss), sd = sd(loss)/sqrt(length(loss))) %>%
  write.csv(file = paste0(img_path, "/risk.csv"), row.names = F)


## RMSE
df %>%
  mutate(loss = (estimate - truth)^2) %>%
  group_by(set, estimator) %>%
  summarise(RMSE = sqrt(mean(loss))) %>%
  write.csv(file = paste0(img_path, "/RMSE.csv"), row.names = F)


# ---- Sampling distributions ----

loadestimates <- function(set, type = "scenarios") {
  df <- read.csv(paste0(int_path, "/estimates_", type, "_", set, ".csv"))
  df$set <- set

  # Relabel to control the order of boxplots
  # df$estimator <- as.factor(df$estimator)
  # levels(df$estimator) <- c("GNN_S", "GNN_Svariable", "GNN_Sclustered", "MAP")
  df$estimator[df$estimator == "GNN_S"] <- "GNN_S1"
  df$estimator[df$estimator == "GNN_Svariable"] <- "GNN_S2"
  df$estimator[df$estimator == "GNN_Sclustered"] <- "GNN_S3"
  df$estimator[df$estimator == "GNN_Smatern"] <- "GNN_S4"

  df
}

loaddata <- function(set) {
  df <- read.csv(paste0(int_path, "/Z_", set, ".csv"))
  df$set <- set
  df
}


# load data
sets <- c("S", "Stilde", "Sclustered")
df  <- lapply(sets, loadestimates); df  <- do.call(rbind, df); df <- filter(df, estimator %in% estimators)
zdf <- lapply(sets, loaddata);      zdf <- do.call(rbind, zdf)

figures <- lapply(unique(df$k), function(K) {

  df  <- df  %>% filter(k == K)
  zdf <- zdf %>% filter(k == K)

  data <- lapply(sets, function(st) {
    field_plot(filter(zdf, set == st), regular = F)
  })
  data[[1]] <- data[[1]] +
    scale_x_continuous(breaks = c(0.25, 0.5, 0.75), expand = c(0, 0)) +
    scale_y_continuous(breaks = c(0.25, 0.5, 0.75), expand = c(0, 0)) +
    labs(fill = "Z") +
    theme(legend.title.align = 0.25, legend.title = element_text(face = "bold"))
  data_legend <- get_legend(data[[1]])

  data[[1]] <- data[[1]] + labs(title = "S")
  data[[2]] <- data[[2]] + labs(title = "S'")
  data[[3]] <- data[[3]] + labs(title = "S'")


  data <- lapply(data, function(gg) gg +
                   theme(legend.position = "none") +
                   theme(plot.title = element_text(hjust = 0.5)) #+ coord_fixed()
  )

  data[-1] <- lapply(data[-1], function(gg) gg +
                       theme(axis.text.y = element_blank(),
                             axis.ticks.y = element_blank(),
                             axis.title.y = element_blank()))


  # ---- Marginal sampling distributions ----

  box <- lapply(sets, function(st) {
    plotdistribution(filter(df, set == st), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1, return_list = TRUE)
  })
  p <- length(unique(df$parameter))
  box_split <- lapply(1:p, function(i) {
    lapply(1:length(box), function(j) box[[j]][[i]])
  })

  # Modify the axes for pretty plotting
  for (i in 1:p) {

    box_split[[i]][[1]] <- box_split[[i]][[1]] + labs(y = box_split[[i]][[1]]$labels$x)

    # Remove axis labels for internal panels
    box_split[[i]][-1] <- lapply(box_split[[i]][-1], function(gg) gg +
                                   theme(axis.text.y = element_blank(),
                                         axis.ticks.y = element_blank(),
                                         axis.title.y = element_blank()))

    # Ensure axis limits are consistent for all panels in a given row (parameter)
    ylims <- df %>% filter(parameter == unique(df$parameter)[i]) %>% summarise(range(estimate)) %>% as.matrix %>% c
    box_split[[i]] <- lapply(box_split[[i]], function(gg) gg + ylim(ylims))
  }

  box <- do.call(c, box_split)
  box <- lapply(box, function(gg) gg + scale_estimator(df))
  box_legend <- get_legend(box[[1]])
  box <- lapply(box, function(gg) {
    gg$facet$params$nrow <- 2
    gg$facet$params$strip.position <- "bottom"
    gg + theme(legend.position = "none", axis.title.x = element_blank()) +scale_estimator(df)
  })




  plotlist <- c(data, box)
  figure1  <- egg::ggarrange(plots = plotlist, nrow = 3, ncol = length(sets), heights = c(1.5, 1, 1))
  figure2  <- ggpubr::ggarrange(data_legend, box_legend, ncol = 1, heights = c(1.2, 2.2))
  figure   <- ggpubr::ggarrange(figure1, figure2, widths = c(1, 0.15))

  ggsave(
    figure,
    file = paste0("samplingdistributions", K, ".pdf"),
    width = 8.5, height = 4.6, device = "pdf", path = img_path
  )

  figure
})
