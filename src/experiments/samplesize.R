library("optparse")
option_list <- list(
  make_option("--model", type="character", default=NULL, metavar="character"),
  make_option("--neighbours", type="character", default="radius", metavar="character")
)
opt_parser  <- OptionParser(option_list=option_list)
model       <- parse_args(opt_parser)$model
neighbours  <- parse_args(opt_parser)$neighbours

int_path <- paste("intermediates/experiments/samplesize", model, neighbours, sep = "/")
img_path <- paste("img/experiments/samplesize", model, neighbours, sep = "/")
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


# Zoom in on the larger sample sizes.
xmin=200;  xmax=400
ymin=0.03; ymax = 0.05

window <- figure +
  theme(axis.title.y = element_blank()) +
  scale_y_continuous(position = "right") +
  coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax))

figure <- figure +
  geom_rect(aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), 
            linewidth = 0.3, colour = "grey50", fill = "transparent")


#TODO Could add another window focusing around small sample sizes (n = 30)

# Extract the legend and convert it to a ggplot object so that it can be added
# to figure in a custom position
legend_plot <- figure %>% get_legend %>% as_ggplot

# Add some padding around window
window <- window + theme(plot.margin = unit(c(10, 10, 20, 10), "points"))

figure <- ggarrange(
  figure,
  ggarrange(legend_plot, window, ncol = 1, legend = "none"),
  legend = "none"
)

ggsave(
  figure,
  file = "risk_vs_n_window.pdf",
  width = 8, height = 4, path = img_path, device = "pdf"
)

