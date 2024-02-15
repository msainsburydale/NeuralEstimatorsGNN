int_path <- file.path("intermediates", "supplement", "neighbours")
img_path <- file.path("img", "supplement", "neighbours")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)
source(file.path("src", "plotting.R"))

library("reshape2")

## Training risk (check for convergence)
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
figure_training # TODO save this plot



## RMSE
df <- read.csv(file.path(int_path, "estimates.csv"))
df <- df %>%
  mutate(loss = (estimate - truth)^2) %>%
  group_by(estimator, n) %>% 
  # group_by(parameter, .add = TRUE) %>% # group by parameter to illustrate that it's not just range parameters that are affected
  summarise(rmse = sqrt(mean(loss)))

breaks <- unique(df$n)
breaks <- breaks[breaks != 60]

# param_labeller <- label_parsed
# df <- mutate_at(df, .vars = "parameter", .funs = factor, levels = names(parameter_labels), labels = parameter_labels)

text_size <- 14

figure_rmse <- ggplot(data = df, aes(x = n, y = rmse, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  # facet_wrap(~parameter, labeller = param_labeller) +
  labs(colour = "", x = expression(n), y = "RMSE") +
  scale_estimator(df) +
  theme_bw(base_size = text_size) +
  theme(
    strip.text.x = element_text(size = 12),
    legend.text.align = 0,
    panel.grid = element_blank(),
    strip.background = element_blank()
  )


## Inference time
df <- read.csv(file.path(int_path, "runtime_singledataset.csv"))

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
  theme_bw(base_size = text_size) +
  guides(colour = guide_legend(byrow = TRUE)) +
  theme(
    legend.text.align = 0,
    panel.grid = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_blank(), 
    legend.spacing.y = unit(1, "lines")
  )

# figure <- egg::ggarrange(figure_rmse + theme(legend.position = "none"), figure_time, nrow = 1, widths = c(1.5, 1))
figure <- egg::ggarrange(figure_rmse + theme(legend.position = "none"), figure_time, nrow = 1)

ggsv(figure, file = "rmse_runtime_vs_n", width = 12, height = 4, path = img_path)
