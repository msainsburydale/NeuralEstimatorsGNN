int_path <- file.path("intermediates", "supplement", "neighbours")
img_path <- file.path("img", "supplement", "neighbours")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)
source(file.path("src", "plotting.R"))
library("reshape2")

# ---- Training risk (check for convergence) ----

all_dirs  <- list.dirs(path = int_path, recursive = F)
df <- lapply(all_dirs, function(dir) {
  loss_per_epoch <- read.csv(file.path(dir, "loss_per_epoch.csv"), header = FALSE)
  colnames(loss_per_epoch) <- c("Training set", "Validation set")
  loss_per_epoch$estimator <- basename(dir)
  loss_per_epoch$epoch <- 0:(nrow(loss_per_epoch) - 1)
  loss_per_epoch
})
df <- do.call("rbind", df)
df <- melt(df, id.vars = c("estimator", "epoch"), variable.name = "set", value.name = "risk")
figure_training <- ggplot(data = df, aes(x = epoch, y = risk, colour = estimator, linetype = set)) +
  geom_line(alpha = 0.75) +
  labs(colour = "", linetype = "", y = "Empirical Bayes risk") +
  coord_cartesian(ylim=c(min(df$risk), 0.15)) +
  scale_estimator(df) +
  theme_bw(base_size = text_size) +
  theme(
    strip.text.x = element_text(size = 12),
    legend.text.align = 0,
    panel.grid = element_blank(),
    strip.background = element_blank()
  )
figure_training
ggsv(figure_training, file = "risk_vs_epoch", width = 7, height = 4, path = img_path)


# ---- All neighbourhood definitions ----

## RMSE
df <- read.csv(file.path(int_path, "estimates.csv")) %>%
  mutate(loss = (estimate - truth)^2) %>%
  group_by(estimator, n) %>%
  summarise(rmse = sqrt(mean(loss)))

rmse_lims <- range(df$rmse)

figure_rmse <- ggplot(data = df, aes(x = n, y = rmse, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  labs(y = "RMSE") +
  scale_estimator(df) +
  theme_bw(base_size = text_size) +
  theme(panel.grid = element_blank())

## Inference time
df <- read.csv(file.path(int_path, "runtime_singledataset.csv"))

time_lims <- range(df$time)

figure_time <- ggplot(data = df,
                      aes(x = n, y = time, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  labs(
    colour = "Neighbourhood",
    x = expression(n),
    y = "Inference time (s)"
  ) +
  scale_estimator(df) +
  theme_bw(base_size = text_size) +
  guides(colour = guide_legend(byrow = TRUE)) +
  theme(
    legend.text.align = 0,
    panel.grid = element_blank(),
    legend.spacing.y = unit(0.5, "lines")
  )

# ---- Maxmin: Sensitivity analysis with respect to k ----

## RMSE
df <- read.csv(file.path(int_path, "k_vs_n.csv")) %>%
  mutate(loss = (estimate - truth)^2) %>%
  group_by(k, n) %>%
  summarise(rmse = sqrt(mean(loss)), time = mean(inference_time))

df$estimator <- ordered(df$k)
rmse_lims <- range(c(rmse_lims, df$rmse))

k_rmse <- ggplot(data = df %>% filter(n <= 1000), aes(x = n, y = rmse, colour = estimator, group = estimator)) +
  geom_point() +
  ylim(rmse_lims) +
  geom_line(alpha = 0.75) +
  labs(colour = "Number of neighbours, k", x = expression(n), y = "RMSE") +
  scale_color_brewer(palette="Reds") +
  theme_bw(base_size = text_size) +
  theme(panel.grid = element_blank())

## Inference time
time_lims <- range(c(time_lims, df$time))
k_time <- ggplot(data = df, aes(x = n, y = time, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  labs(colour = "k", y = "Inference time (s)") +
  ylim(time_lims) +
  theme_bw(base_size = text_size) +
  scale_color_brewer(palette="Reds") +
  theme(panel.grid = element_blank())

# ---- Combined plot ----

figure <- egg::ggarrange(
  figure_rmse + theme(legend.position = "none"), figure_time,
  k_rmse + theme(legend.position = "none"), k_time,
  nrow = 2)

ggsv(figure, file = "neighbourhood", width = 9.4, height = 5.3, path = img_path)
