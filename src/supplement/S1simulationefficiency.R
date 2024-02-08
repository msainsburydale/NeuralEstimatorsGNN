int_path <- file.path("intermediates", "supplement", "simulationefficiency")
img_path <- file.path("img", "supplement", "simulationefficiency")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)
source(file.path("src", "plotting.R"))

df <- read.csv(file.path(int_path, "assessment.csv"))
df <- df %>%
  mutate(loss1 = abs(estimate - truth), loss2 = (estimate - truth)^2) %>%
  group_by(estimator, K) %>% 
  # group_by(parameter, .add = TRUE) %>% # group by parameter to illustrate that it's not just range parameters that are affected
  summarise(rmse = sqrt(mean(loss2)), bayes_risk = mean(loss1))

text_size <- 14

figure_rmse <- ggplot(data = df, aes(x = K, y = rmse, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  labs(x = "Number of simulated data sets", y = "RMSE") +
  scale_estimator(df) +
  theme_bw(base_size = text_size) +
  theme(
    strip.text.x = element_text(size = 12),
    legend.text.align = 0,
    panel.grid = element_blank(),
    strip.background = element_blank()
  )

figure_risk <- ggplot(data = df, aes(x = K, y = bayes_risk, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  labs(colour = expression(paste("Prior for ", bold(S))), x = "Number of simulated data sets", y = "Bayes risk") +
  scale_estimator(df) +
  theme_bw(base_size = text_size) +
  theme(
    strip.text.x = element_text(size = 12),
    legend.text.align = 0,
    panel.grid = element_blank(),
    strip.background = element_blank()
  )

figure <- egg::ggarrange(figure_rmse + theme(legend.position = "none"), figure_risk, nrow = 1)

ggsave(
  figure,
  file = "simulation_efficiency.pdf",
  width = 10.9, height = 4, path = img_path, device = "pdf"
)
