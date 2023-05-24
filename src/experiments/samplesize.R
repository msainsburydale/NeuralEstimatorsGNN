library("optparse")
option_list <- list(
  make_option("--model", type="character", default=NULL, metavar="character")
)
opt_parser  <- OptionParser(option_list=option_list)
model       <- parse_args(opt_parser)$model

int_path <- paste0("intermediates/experiments/samplesize/", model)
img_path <- paste0("img/experiments/samplesize/", model)
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

source("src/plotting.R")


# ---- Risk function ----

df <- read.csv(paste0(int_path, "/estimates_test.csv"))

## Bayes risk with respect to absolute error
df <- df %>%
  mutate(loss = abs(estimate - truth)) %>% 
  group_by(estimator, n) %>% 
  summarise(risk = mean(loss), sd = sd(loss)/sqrt(length(loss)))

ggplot(df) + 
  geom_line(aes(x = n, y = risk, colour = estimator)) + 
  theme_bw()


# ---- average risk plot ----

breaks <- unique(df$n)
breaks <- breaks[breaks != 60]

figure <- ggplot(data = df,
                 aes(x = n, y = risk, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  labs(colour = "", x = expression(n), y = expression(r[Omega](hat(theta)("· ; ·")))) +
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

# Zoom in on the larger sample sizes.
xmin=200;  xmax=400
ymin=0.03; ymax = 0.05

window <- figure +
  theme(
    # axis.title.x = element_blank(),
    axis.title.y = element_blank()
  ) +
  scale_y_continuous(position = "right") +
  coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax))


figure <- figure +
  geom_rect(aes(xmin=xmin, xmax=xmax + 2, ymin=ymin, ymax=ymax), 
            linewidth = 0.3, colour = "grey50", fill = "transparent")

# Extract the legend and convert it to a ggplot object so that it can be added
# to figure in a custom position
legend_plot <- figure %>% get_legend %>% as_ggplot

# Add some padding around window
window <- window + theme(plot.margin = unit(c(10, 10, 20, 10), "points"))

g <- ggarrange(
  figure,
  ggarrange(legend_plot, window, ncol = 1, legend = "none"),
  legend = "none"
)

g

ggsave(
  g,
  file = "risk_vs_n.pdf",
  width = 8, height = 4, path = img_path, device = "pdf"
)

