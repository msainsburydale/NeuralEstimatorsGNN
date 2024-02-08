img_path <- file.path("img", "supplement")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)
source(file.path("src", "plotting.R"))

text_size <- 14

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

figure1 <- ggplot(data = df, aes(x = n, y = rmse, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  labs(colour = "", y = TeX("RMSE under $\\bf{S} \\sim $UBPP(n)")) +
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
       y = TeX("RMSE under $\\bf{S} = \\bf{S}_0$"), 
       # colour = expression(paste("Prior for ", bold(S)))
       # colour = expression(paste("Prior for ", bold(S), "\n during training"))
       colour = ""
       )+
  scale_estimator(df) +
  guides(colour = guide_legend(byrow = TRUE)) +
  theme_bw(base_size = text_size) +
  theme(legend.text.align = 0, panel.grid = element_blank(), legend.spacing.y = unit(2, "lines"))

ggsave(
  figure2,
  file = "simulation_efficiency.pdf",
  width = 7.2, height = 3.8, path = img_path, device = "pdf"
)

figure2 <- figure2 + labs(colour = "") + theme(legend.position = "top")




# ---- Combine and save ----

figure <- egg::ggarrange(figure1, figure2, nrow = 1)

ggsave(
  figure,
  file = "prior_for_S.pdf",
  width = 8.2, height = 4.1, path = img_path, device = "pdf"
)



