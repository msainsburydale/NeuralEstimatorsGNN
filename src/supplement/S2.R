img_path <- file.path("img", "supplement")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)
source(file.path("src", "plotting.R"))

# ---- Variable sample sizes ----

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
  labs(colour = "", y = TeX("RMSE under $S \\sim $UBPP(n)")) +
  scale_x_continuous(breaks = breaks) +
  scale_estimator(df) +
  theme_bw(base_size = text_size) +
  theme(legend.text.align = 0, legend.position = "top", panel.grid = element_blank()) 

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
       y = TeX("RMSE under $S = S_0$"), 
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



