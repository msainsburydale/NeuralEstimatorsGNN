img_path <- file.path("img", "supplement")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)
source(file.path("src", "plotting.R"))

# ---- Variable sample sizes: point estimator ----

int_path <- file.path("intermediates", "supplement", "variablesamplesize")

df <- read.csv(file.path(int_path, "estimates.csv"))
df <- filter(df, estimator != "ML")

df <- df %>%
  mutate(loss = (estimate - truth)^2) %>%
  group_by(estimator, n) %>%
  summarise(rmse = sqrt(mean(loss)))

breaks <- unique(df$n)
breaks <- breaks[breaks != 60]
breaks <- breaks[breaks != 100]
breaks <- breaks[breaks != 350]

figure1 <- ggplot(data = df, aes(x = n, y = rmse, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  labs(colour = "", y = TeX("RMSE under $\\bf{S} \\sim $UBPP(n)")) +
  scale_x_continuous(breaks = breaks) +
  scale_estimator(df) +
  theme_bw(base_size = text_size) +
  theme(legend.text.align = 0, legend.position = "top", panel.grid = element_blank()) 

# Zoom in on the larger sample sizes
figure <- figure1
xlim1 <- c(900, 1100)
ylim1 <- c(0.01, 0.055)
window1 <- figure +
  theme(axis.title.y = element_blank()) +
  scale_y_continuous(position = "right") +
  coord_cartesian(xlim = xlim1, ylim = ylim1) + 
  theme(aspect.ratio = .5)

# Zoom in on the smaller sample sizes
GNN1_RMSE <- df$rmse[df$estimator == "GNN1" & df$n == 30]
xlim <- c(15, 45)
ylim <- GNN1_RMSE + c(-0.15, 0.15)*GNN1_RMSE
window2 <- figure +
  theme(axis.title.y = element_blank()) +
  scale_y_continuous(position = "right") +
  coord_cartesian(xlim = xlim, ylim = ylim) + 
  theme(aspect.ratio = 2)

# Add some padding to the windows
window1 <- window1 + theme(plot.margin = unit(c(10, 10, 20, 10), "points"))
window2 <- window2 + theme(plot.margin = unit(c(60, 5, 5, 5), "points"))

# Add a gray box to the main figure indicating the windows
figure <- figure +
  geom_rect(aes(xmin=xlim1[1], xmax=xlim1[2], ymin=ylim1[1], ymax=ylim1[2]),
            linewidth = 0.3, colour = "grey50", fill = "transparent") +
  geom_rect(aes(xmin=xlim[1], xmax=xlim[2], ymin=ylim[1], ymax=ylim[2]),
            linewidth = 0.3, colour = "grey50", fill = "transparent")

figure <- ggpubr::ggarrange(
  figure,
  ggpubr::ggarrange(window2, window1, ncol = 1, legend = "none"),
  legend = "top", 
  nrow = 1
)

ggsv("variable_sample_size", figure, path = img_path, width = 9.5, height = 4.5)


# ---- Variable sample sizes: point estimator ----

df <- read.csv(file.path(int_path, "estimates_interval.csv"))

a <- df$α[1]

df <- df %>%
  filter(!(n %in% c(60, 100))) %>% 
  mutate(
    length = upper - lower, 
    within = lower <= truth & truth <= upper, 
    IS = length + (2/a) * (lower - truth) * (truth < lower) + (2/a) * (truth - upper) * (truth > upper)
    ) %>%
  group_by(estimator, n, parameter) %>%
  summarise(length = mean(length), coverage = mean(within), IS = mean(IS)) %>% 
  melt(measure.vars = c("coverage", "length", "IS"))

df2 <- data.frame(variable = "coverage", parameter = unique(df$parameter), yintercept = 1-a)

df  <- mutate_at(df, .vars = "parameter", .funs = factor, levels = names(parameter_labels), labels = parameter_labels)
df2 <- mutate_at(df2, .vars = "parameter", .funs = factor, levels = names(parameter_labels), labels = parameter_labels)

figure <- ggplot(data = df, aes(x = n, y = value)) +
  geom_point() +
  geom_line() +
  geom_hline(data=df2, aes(yintercept=yintercept), colour="red", linetype = "dashed") + 
  labs(colour = "", y = "") +
  ylim(c(0.1, NA)) + 
  facet_grid(variable ~ parameter, labeller = label_parsed) + 
  theme_bw(base_size = text_size) +
  theme(panel.grid = element_blank()) 

ggsv("variable_sample_size_interval", figure, path = img_path, width = 7.6, height = 5.8)


# ---- Simulation efficiency ----

int_path <- file.path("intermediates", "supplement", "simulationefficiency")

df <- read.csv(file.path(int_path, "assessment.csv"))
df <- df %>%
  mutate(loss = (estimate - truth)^2) %>%
  group_by(estimator, K) %>% 
  summarise(rmse = sqrt(mean(loss)))

df <- df %>% filter(estimator %in% c("Sfixed", "Srandom_cluster"))

figure2 <- ggplot(data = df, aes(x = K, y = rmse, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  labs(x = "Number of simulated data sets", 
       y = TeX("RMSE under $\\bf{S} = \\bf{S}_0$"), 
       colour = "")+
  scale_estimator(df) +
  guides(colour = guide_legend(byrow = TRUE)) +
  theme_bw(base_size = text_size) +
  theme(legend.text.align = 0, 
        panel.grid = element_blank(), 
        legend.spacing.y = unit(2, "lines"), 
        legend.position = "top")


# ---- Combine and save ----

figure <- egg::ggarrange(figure1, figure2, nrow = 1)

ggsv(figure, file = "prior_for_S", width = 8.2, height = 4.1, path = img_path)



