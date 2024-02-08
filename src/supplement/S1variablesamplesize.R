int_path <- file.path("intermediates", "supplement", "variablesamplesize")
img_path <- file.path("img", "supplement", "variablesamplesize")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

source(file.path("src", "plotting.R"))

df <- read.csv(file.path(int_path, "estimates.csv"))
df <- filter(df, estimator != "ML")

## RMSE plot

df <- df %>%
  mutate(loss = (estimate - truth)^2) %>%
  group_by(estimator, n) %>%
  summarise(rmse = sqrt(mean(loss)))


breaks <- unique(df$n)
breaks <- breaks[breaks != 60]
breaks <- breaks[breaks != 100]

figure <- ggplot(data = df, aes(x = n, y = rmse, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  labs(colour = "", y = TeX("RMSE under $\\bf{S} \\sim $UBPP(n)")) +
  scale_x_continuous(breaks = breaks) +
  scale_estimator(df) +
  theme_bw(base_size = text_size) +
  theme(legend.text.align = 0, panel.grid = element_blank()) 

figure_RMSE <- figure

# Zoom in on the larger sample sizes.
xmin1=900;  xmax1=1100
ymin1=0.01; ymax1 = 0.055

window1 <- figure +
  theme(axis.title.y = element_blank()) +
  scale_y_continuous(position = "right") +
  coord_cartesian(xlim = c(xmin1, xmax1), ylim = c(ymin1, ymax1)) + 
  theme(aspect.ratio = .5)

#TODO the position of the windows are fixed, and so in this case, one of the 
#  windows does not contain the curves for any of the estimators. Could this 
#  potentially still be the case when running the non-fast option? Would it 
#  make sense to centre the window around the value at (n=30) for the Bayes 
#  estimator, rather than having it fixed? Yes, it would...

# Zoom in on the smaller sample sizes.
xmin=15;  xmax=45
ymin=0.105; ymax = 0.135

window2 <- figure +
  theme(axis.title.y = element_blank()) +
  scale_y_continuous(position = "right") +
  coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax)) + 
  theme(aspect.ratio = 2)

# Add some padding to the windows
window1 <- window1 + theme(plot.margin = unit(c(10, 10, 20, 10), "points"))
window2 <- window2 + theme(plot.margin = unit(c(60, 5, 5, 5), "points"))

# Add a gray box to the main figure indicating the windows
figure <- figure +
  geom_rect(aes(xmin=xmin1, xmax=xmax1, ymin=ymin1, ymax=ymax1),
            linewidth = 0.3, colour = "grey50", fill = "transparent") +
  geom_rect(aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax),
            linewidth = 0.3, colour = "grey50", fill = "transparent")

suppressWarnings({
  figure <- ggpubr::ggarrange(
    figure,
    ggpubr::ggarrange(window2, window1, ncol = 1, legend = "none"),
    legend = "top", 
    nrow = 1
  )
})

suppressWarnings({
  ggsave(
    figure,
    file = "rmse_vs_n_window.pdf",
    width = 9.5, height = 4.5, path = img_path, device = "pdf"
  )
})


# ---- Timing plot ----

df <- read.csv(file.path(int_path, "runtime.csv"))

average_nbes <- F
if (average_nbes) {
  df$estimator[df$estimator != "ML"] <- "NBE"
  df <- df %>% group_by(estimator, n) %>% summarise(time = mean(time))
  estimator_colours <- c(estimator_colours, "NBE" = estimator_colours[["GNN3"]])
  estimator_labels  <- c(estimator_labels, "NBE" = "NBE")
  estimator_order <- names(estimator_labels) 
}

figure_time <- ggplot(data = df,
                      aes(x = n, y = time, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  labs(
    colour = "", 
    x = expression(n), 
    y = "Inference time (s)"
  ) +
  # scale_y_continuous(trans='log2') +
  # scale_x_continuous(breaks = breaks) +
  scale_estimator(df) +
  theme_bw() +
  theme(
    legend.text=element_text(size = 12),
    legend.text.align = 0,
    panel.grid = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_blank()
  )

figure_combined <- ggpubr::ggarrange(
  figure_RMSE, 
  figure_time, 
  nrow = 1, 
  common.legend = !average_nbes, 
  legend = "right"
)

ggsave(
  figure_combined,
  file = "RMSE_runtime.pdf",
  width = 9.5, height =3.7, path = img_path, device = "pdf"
)
