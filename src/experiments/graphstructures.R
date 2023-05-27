library("optparse")
option_list <- list(
  make_option("--model", type="character", default=NULL, metavar="character"),
  make_option("--neighbours", type="character", default="radius", metavar="character")
)
opt_parser  <- OptionParser(option_list=option_list)
model       <- parse_args(opt_parser)$model
neighbours  <- parse_args(opt_parser)$neighbours

int_path <- paste("intermediates/experiments/graphstructures", model, neighbours, sep = "/")
img_path <- paste("img/experiments/graphstructures", model, neighbours, sep = "/")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

loadestimates <- function(set, type = "scenarios") {
  df <- read.csv(paste0(int_path, "/estimates_", type, "_", set, ".csv"))
  df$set <- set
  df
}

source("src/plotting.R")

boldtheta <- bquote(bold(theta))
estimators <- names(estimator_labels)

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

df <- loadestimates("S") %>%
  rbind(loadestimates("Stilde")) %>%
  rbind(loadestimates("Sclustered")) %>% 
  filter(estimator %in% estimators)

zdf <- loaddata("S") %>%
  rbind(loaddata("Stilde")) %>%
  rbind(loaddata("Sclustered"))

figures <- lapply(unique(df$k), function(K) {
  
  df  <- df  %>% filter(k == K)
  zdf <- zdf %>% filter(k == K)
  
  # #TODO histogram of spatial distances 
  # hdf <- zdf %>% 
  #   group_by(set) %>% 
  #   summarise(h = c(dist(matrix(c(s1, s2), ncol = 2, byrow = F))))
  # 
  # if (neighbours == "radius") {
  #   hdf <- filter(hdf, h < 0.15)
  # } else {
  #   #TODO for fixednum approach, need to know which nodes are neighbours
  # }
  # 
  # # TODO need to change the ordering here
  # ggh <- ggplot(hdf) + 
  #   geom_histogram(aes(x = h)) + 
  #   facet_wrap(~set)
  # 
  # x <- split(zdf, zdf$set)
  # x <- lapply(x, function(df) {
  #   S = df[, c("s1", "s2")]
  #   S = expand.grid(S)
  #   D = apply(S, 1, function(x) sqrt(sum(x^2)))
  #   df = data.frame(D = D, set = )  
  # })
  
  
  ggz_1  <- field_plot(filter(zdf, set == "S"), regular = F) + labs(title = "S")
  ggz_2  <- field_plot(filter(zdf, set == "Stilde"), regular = F) + labs(title = "S'")
  ggz_3  <- field_plot(filter(zdf, set == "Sclustered"), regular = F) + labs(title = "S''")
  ggz_1  <- ggz_1 + labs(fill = "Z") + theme(legend.title.align=0.25, legend.title = element_text(face = "bold"))
  data_legend <- get_legend(ggz_1)
  data <- list(ggz_1, ggz_2, ggz_3)
  data <- lapply(data, function(gg) gg + 
                   theme(legend.position = "none") + 
                   theme(plot.title = element_text(hjust = 0.5)) + 
                   coord_fixed())
  
  
  box_1  <- plotdistribution(filter(df, set == "S"), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1) # + scale_estimator(df)
  box_2   <- plotdistribution(filter(df, set == "Stilde"), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1) # + scale_estimator(df)
  box_3   <- plotdistribution(filter(df, set == "Sclustered"), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1) # + scale_estimator(df)
  box_legend <- get_legend(box_2)
  box <- list(box_1, box_2, box_3)
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
  figure1  <- ggarrange(plotlist = plotlist, nrow = 2, ncol = 3, heights = c(1.25, 2))
  figure2  <- ggarrange(data_legend, box_legend, ncol = 1, heights = c(1, 2.5))
  figure   <- ggarrange(figure1, figure2, widths = c(1, 0.15))
  figure
  
  ggsave(
    figure, 
    file = paste0("samplingdistributions", K, ".pdf"),
    width = 8.5, height = 6, device = "pdf", path = img_path
  )
  
  figure
})




