model       <- "GP/nuFixed"
int_path <- paste("intermediates/supplement/variablesamplesize", model, sep = "/")
img_path <- paste("img/supplement/variablesamplesize", model, sep = "/")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

source("src/plotting.R")

df <- read.csv(paste0(int_path, "/estimates.csv"))

## Bayes risk with respect to absolute error
# df <- df %>%
#   mutate(loss = abs(estimate - truth)) %>%
#   group_by(estimator, n) %>%
#   summarise(risk = mean(loss), sd = sd(loss)/sqrt(length(loss)))

## RMSE
df <- df %>%
  mutate(loss = (estimate - truth)^2) %>%
  group_by(estimator, n) %>%
  summarise(risk = sqrt(mean(loss))) 

## average risk plot
breaks <- unique(df$n)
breaks <- breaks[breaks != 60]
breaks <- breaks[breaks != 100]

figure <- ggplot(data = df,
                 aes(x = n, y = risk, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  labs(
    colour = "", 
    x = expression(n), 
    # y = expression(r[Omega](hat(theta)("·")))
    y = "RMSE"
  ) +
  scale_x_continuous(breaks = breaks) +
  scale_estimator(df) +
  theme_bw() +
  # coord_cartesian(ylim=c(0, 0.35)) + 
  theme(
    legend.text=element_text(size = 14),
    legend.text.align = 0,
    panel.grid = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_blank()
  )

suppressWarnings({
  ggsave(
    figure,
    file = "risk_vs_n.pdf",
    width = 8, height = 4, path = img_path, device = "pdf"
  )
})

# Zoom in on the larger sample sizes.
xmin1=900;  xmax1=1100
ymin1=0.01; ymax1 = 0.055

window1 <- figure +
  theme(axis.title.y = element_blank()) +
  scale_y_continuous(position = "right") +
  coord_cartesian(xlim = c(xmin1, xmax1), ylim = c(ymin1, ymax1)) + 
  theme(aspect.ratio = .5)

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
    file = "risk_vs_n_window.pdf",
    width = 9.5, height = 4.5, path = img_path, device = "pdf"
  )
})


# ---- Timing plot ----

# TODO 

df <- read.csv(paste0(int_path, "/runtime.csv"))

figure <- ggplot(data = df,
                 aes(x = n, y = time, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  labs(
    colour = "", 
    x = expression(n), 
    y = "Time (s)"
  ) +
  scale_x_continuous(breaks = breaks) +
  scale_estimator(df) +
  theme_bw() +
  theme(
    legend.text=element_text(size = 14),
    legend.text.align = 0,
    panel.grid = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_blank()
  )
