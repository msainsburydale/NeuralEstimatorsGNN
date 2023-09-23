library("optparse")
option_list <- list(
  make_option("--model", type="character", default=NULL, metavar="character")
)
opt_parser  <- OptionParser(option_list=option_list)
model       <- parse_args(opt_parser)$model

int_path <- paste0("intermediates/", model)
img_path <- paste0("img/", model)
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

source("src/plotting.R")

loadestimates <- function(set, type = "scenarios") {
  df <- read.csv(paste0(int_path, "/estimates_", type, "_", set, ".csv"))
  df$set <- set
  df
}

loadcoverage <- function(set) {
  df <- read.csv(paste0(int_path, "/conditionalcoverage_", set, ".csv"))
  df$set <- set
  p <- length(unique(df$parameter))
  df$pair <- rep(1:(nrow(df)/p), each = p)
  df
}

loaddata <- function(set) {
  df <- read.csv(paste0(int_path, "/Z_", set, ".csv"))
  df$set <- set
  df
}

estimators <- c("GNN", "ML")

if (model %in% c("Schlather", "BrownResnick")) {
  estimator_labels["ML"] <- "PL"
}

# ---- Marginal + joint sampling distribution with uniform spatial locations ----

df  <- loadestimates("uniform", "scenarios") %>% filter(estimator %in% estimators)
zdf <- loaddata("uniform")
p <- length(unique(df$parameter))

if (p == 2) {
  figures <- lapply(unique(df$k), function(K) {
    
    df <- df %>% filter(k == K)
    zdf <- zdf %>% filter(k == K)
    
    gg1 <- field_plot(zdf, regular = F) #+ coord_fixed()
    gg2 <- plotdistribution(df, type = "scatter", parameter_labels = parameter_labels)[[1]] # , truth_line_size = 1
    gg3 <- plotdistribution(df, parameter_labels = parameter_labels, return_list = T)
    
    suppressMessages({
      gg2 <- gg2 + scale_estimator(df)
      gg3 <- lapply(gg3, function(gg) gg + scale_estimator(df) + theme(legend.position = "top") + labs(y = ""))
    })
    
    
    gg1 <- gg1 + theme(legend.position = "top", legend.title.align = 0.5, legend.title = element_text(face = "bold"))
    gg2 <- gg2 + theme(legend.position = "top")
    
    figure <- ggpubr::ggarrange(plotlist = c(list(gg1, gg2), gg3), nrow = 1, ncol = 4, align = "hv")
    
    ggsave(
      figure,
      file = paste0("uniformlocations", K, ".pdf"),
      width = 9.3, height = 3, device = "pdf", path = img_path
    )
    
    figure
  }) 
}

# ---- Risk function ----

sets <- c("uniform", "quadrants", "mixedsparsity", "cup")
df <- lapply(sets, loadestimates, type = "test")
df <- do.call(rbind, df)
df <- filter(df, estimator %in% estimators)

## Bayes risk with respect to absolute error
df %>%
  mutate(loss = abs(estimate - truth)) %>%
  group_by(estimator) %>%
  summarise(risk = mean(loss), sd = sd(loss)/sqrt(length(loss))) %>%
  write.csv(file = paste0(img_path, "/risk.csv"), row.names = F)


## RMSE
df %>%
  mutate(loss = (estimate - truth)^2) %>%
  group_by(estimator) %>%
  summarise(RMSE = sqrt(mean(loss))) %>%
  write.csv(file = paste0(img_path, "/RMSE.csv"), row.names = F)


# ---- Sampling distributions ----

# load data
sets <- c("uniform", "quadrants", "mixedsparsity", "cup")
df  <- lapply(sets, loadestimates); df  <- do.call(rbind, df); df <- filter(df, estimator %in% estimators)
zdf <- lapply(sets, loaddata);      zdf <- do.call(rbind, zdf)

figures <- lapply(unique(df$k), function(K) {
  
  df  <- df  %>% filter(k == K)
  zdf <- zdf %>% filter(k == K)
  
  data <- lapply(sets, function(st) {
    field_plot(filter(zdf, set == st), regular = F)
  })
  suppressMessages({
    data[[1]] <- data[[1]] +
      scale_x_continuous(breaks = c(0.25, 0.5, 0.75), expand = c(0, 0)) +
      scale_y_continuous(breaks = c(0.25, 0.5, 0.75), expand = c(0, 0)) +
      labs(fill = "Z") +
      theme(legend.title.align = 0.25, legend.title = element_text(face = "bold"))
  })
  data_legend <- get_legend(data[[1]])
  
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
  suppressMessages({
    box <- lapply(box, function(gg) gg + scale_estimator(df))
  })
  box_legend <- get_legend(box[[1]])
  suppressMessages({
    box <- lapply(box, function(gg) {
      gg$facet$params$nrow <- 2
      gg$facet$params$strip.position <- "bottom"
      gg + theme(legend.position = "none", axis.title.x = element_blank()) +scale_estimator(df)
    })
  })
  
  plotlist <- c(data, box)
  figure1  <- egg::ggarrange(plots = plotlist, nrow = p + 1, ncol = length(sets), heights = c(1.5, rep(1, p)))
  figure2  <- ggpubr::ggarrange(data_legend, box_legend, ncol = 1, heights = c(1.2, 2.2))
  figure   <- ggpubr::ggarrange(figure1, figure2, widths = c(1, 0.15))
  
  ggsave(
    figure,
    file = paste0("samplingdistributions_marginal", K, ".pdf"),
    width = 8.5, height = 4.6, device = "pdf", path = img_path
  )
  
  # ---- Joint sampling distributions ----
  
  if (p == 2) {
    suppressMessages({
      joint <- lapply(sets, function(st) {
        plotdistribution(filter(df, set == st), type = "scatter", parameter_labels = parameter_labels, estimator_labels = estimator_labels)[[1]] + scale_estimator(df)
      })
    })
    joint_legend <- get_legend(joint[[1]])
    joint <- lapply(joint, function(gg) {
      gg$facet$params$nrow <- 2
      gg$facet$params$strip.position <- "bottom"
      gg + theme(legend.position = "none", strip.background = element_blank())
    })
    joint[-1] <- lapply(joint[-1], function(gg) gg +
                          theme(axis.text.y = element_blank(),
                                axis.ticks.y = element_blank(),
                                axis.title.y = element_blank()))
    
    # limits:
    g <- joint[[1]]
    xname <- quo_name(g$layers[[1]]$mapping$x); xname <- gsub("estimate_", "", xname)
    yname <- quo_name(g$layers[[1]]$mapping$y); yname <- gsub("estimate_", "", yname)
    xlims <- df %>% filter(parameter == xname) %>% summarise(range(estimate)) %>% as.matrix %>% c
    ylims <- df %>% filter(parameter == yname) %>% summarise(range(estimate)) %>% as.matrix %>% c
    joint <- lapply(joint, function(gg) gg + xlim(xlims) + ylim(ylims))
    
    plots <- c(data, joint)
    figure1  <- egg::ggarrange(plots = plots, nrow = 2, ncol = length(sets), align = "hv")
    figure2  <- ggpubr::ggarrange(data_legend, joint_legend, ncol = 1, heights = c(1, 1))
    figure   <- ggpubr::ggarrange(figure1, figure2, widths = c(1, 0.15))
    
    ggsave(
      figure,
      file = paste0("samplingdistributions_joint", K, ".pdf"),
      width = 8, height = 4, device = "pdf", path = img_path
    )
  }
  
  figure
})



# ---- Conditional coverage ----

# # load data
# sets <- c("uniform", "quadrants", "mixedsparsity", "cup")
# df  <- lapply(sets, loadcoverage); df  <- do.call(rbind, df)
# zdf <- lapply(sets, loaddata); zdf <- do.call(rbind, zdf)
# zdf <- zdf %>% filter(k == 1)
#
# df <- df %>%
#   pivot_wider(names_from = "parameter", values_from = c("parameter_value", "coverage")) %>%
#   as.data.frame
#
# ggplot(df) +
#   geom_tile(aes(x = parameter_value_τ, y = parameter_value_ρ, fill = coverage_τ)) +
#   scale_fill_viridis_c(option = "magma") +
#   labs(x = expression(tau), y = expression(rho)) +
#   facet_grid( .~ set)
#
# ggplot(df) +
#   geom_tile(aes(x = parameter_value_τ, y = parameter_value_ρ, fill = coverage_ρ)) +
#   scale_fill_viridis_c(option = "magma") +
#   labs(x = expression(tau), y = expression(rho)) +
#   facet_grid(.~ set)
#
