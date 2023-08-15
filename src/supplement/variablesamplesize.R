model       <- "GP/nuFixed"
int_path <- paste("intermediates/supplement/variablesamplesize", model, sep = "/")
img_path <- paste("img/supplement/variablesamplesize", model, sep = "/")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

source("src/plotting.R")

df <- read.csv(paste0(int_path, "/estimates_test.csv"))

## Bayes risk with respect to absolute error
df <- df %>%
  mutate(loss = abs(estimate - truth)) %>%
  group_by(estimator, n) %>%
  summarise(risk = mean(loss), sd = sd(loss)/sqrt(length(loss)))

## average risk plot
breaks <- unique(df$n)
breaks <- breaks[breaks != 60]

figure <- ggplot(data = df,
                 aes(x = n, y = risk, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  labs(colour = "", x = expression(n), y = expression(r[Omega](hat(theta)("·")))) +
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

ggsave(
  figure,
  file = "risk_vs_n.pdf",
  width = 8, height = 4, path = img_path, device = "pdf"
)



# Extract the legend and convert it to a ggplot object so that it can be added
# to figure in a custom position
legend_plot <- figure %>% get_legend %>% as_ggplot

# Zoom in on the larger sample sizes.
xmin1=200;  xmax1=400
ymin1=0.03; ymax1 = 0.05

window1 <- figure +
  theme(axis.title.y = element_blank()) +
  scale_y_continuous(position = "right") +
  coord_cartesian(xlim = c(xmin1, xmax1), ylim = c(ymin1, ymax1))

# Zoom in on the smaller sample sizes.
xmin=15;  xmax=45
ymin=0.09; ymax = 0.12

window2 <- figure +
  theme(axis.title.y = element_blank()) +
  scale_y_continuous(position = "right") +
  coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax))

# Add some padding to the windows
window1 <- window1 + theme(plot.margin = unit(c(10, 10, 20, 10), "points")) 
window2 <- window2 + theme(plot.margin = unit(c(10, 10, 20, 10), "points")) 

ggpubr::ggarrange(window2, window1, ncol = 1, legend = "none", widths = c(0.2, 1))
cowplot::plot_grid(window2, window1, ncol = 1, rel_widths = c(0.2, 1))

# Add a gray box to the main figure indicating the windows
figure <- figure +
  geom_rect(aes(xmin=xmin1, xmax=xmax1, ymin=ymin1, ymax=ymax1),
            linewidth = 0.3, colour = "grey50", fill = "transparent") +
  geom_rect(aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax),
            linewidth = 0.3, colour = "grey50", fill = "transparent")


ggpubr::ggarrange(
  figure,
  ggpubr::ggarrange(legend_plot, window1, ncol = 1, legend = "none"),
  legend = "none"
)


ggpubr::ggarrange(
  figure,
  ggpubr::ggarrange(window2, window1, ncol = 1, legend = "none", widths = c(0.2, 1)),
  legend = "none"
)





figure <- ggpubr::ggarrange(
  figure,
  ggpubr::ggarrange(legend_plot, window, ncol = 1, legend = "none"),
  legend = "none"
)

ggsave(
  figure,
  file = "risk_vs_n_window.pdf",
  width = 8, height = 4, path = img_path, device = "pdf"
)



# # Zoom in on the larger sample sizes.
# xmin=200;  xmax=400
# ymin=0.03; ymax = 0.05
# 
# window <- figure +
#   theme(axis.title.y = element_blank()) +
#   scale_y_continuous(position = "right") +
#   coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax))
# 
# figure <- figure +
#   geom_rect(aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax),
#             linewidth = 0.3, colour = "grey50", fill = "transparent")
# 
# # Extract the legend and convert it to a ggplot object so that it can be added
# # to figure in a custom position
# legend_plot <- figure %>% get_legend %>% as_ggplot
# 
# # Add some padding around window
# window <- window + theme(plot.margin = unit(c(10, 10, 20, 10), "points"))
# 
# figure <- ggarrange(
#   figure,
#   ggarrange(legend_plot, window, ncol = 1, legend = "none"),
#   legend = "none"
# )
# 
# ggsave(
#   figure,
#   file = "risk_vs_n_window.pdf",
#   width = 8, height = 4, path = img_path, device = "pdf"
# )
