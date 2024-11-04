int_path <- file.path("intermediates", "supplement", "neighbours")
img_path <- file.path("img", "supplement", "neighbours")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)
source(file.path("src", "plotting.R"))
library("reshape2")
library("dplyr")
library("ggh4x")
library("scales")

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

df$method <- sub("\\_.*", "", df$estimator)

df <- melt(df, id.vars = c("estimator", "epoch", "method"), variable.name = "set", value.name = "risk")

figure_training <- ggplot(
    data = df %>% filter(epoch < 50), 
    aes(x = epoch, y = risk, colour = estimator, linetype = set)
  ) +
  geom_line(alpha = 0.75) +
  facet_wrap(vars(method)) + 
  labs(colour = "", linetype = "", y = "Empirical Bayes risk") +
  coord_cartesian(ylim=c(min(df$risk), 0.15)) +
  theme_bw(base_size = text_size) +
  theme(
    strip.text.x = element_text(size = 12),
    legend.text.align = 0,
    strip.background = element_blank()
  )

ggsv(figure_training, file = "risk_vs_epoch", width = 10.6, height = 4.5, path = img_path)


# ---- Final figure ----

## Sensitivity analysis: as a function of hyperparameter

df <- read.csv(file.path(int_path, "sensitivity_analysis.csv"))
df <- df %>% 
  filter(n == 250, is.na(k) | (1 < k), is.na(r) | r > 0.01) %>% # TODO remove some of these conditions after rerunning 
  mutate(loss = (estimate - truth)^2) %>%
  group_by(k, r, estimator, n) %>%
  summarise(rmse = sqrt(mean(loss)), time = mean(inference_time)) %>% 
  rowwise() %>% 
  mutate(hyperparameter = max(k, r, na.rm=TRUE), radius_only = !is.na(k)) %>%
  melt(measure.vars = c("rmse", "time"))

rmse_lims <- c(0.01, 0.07)
time_lims <- c(0.001, 0.0035)

axis_labels <- c("rmse" = "RMSE", "time" = "Inference time (s)")

hyperparams <- ggplot(df, aes(x = hyperparameter, y = value, group = estimator, colour = estimator)) +
  geom_point() + 
  geom_line() + 
  facet_grid(variable ~ radius_only, scales = "free", switch = "y", labeller = as_labeller(axis_labels)) +
  scale_estimator(df) +
  theme_bw(base_size = text_size) +
  labs(x = "Hyperparameter") + 
  theme(
    axis.title.y = element_blank(),
    strip.text.x = element_blank(),
    strip.text.y = element_text(size = text_size),
    strip.background = element_blank(),
    legend.position = "none", 
    strip.placement = "outside"
  ) + 
  ggh4x::facetted_pos_scales(y = list(
  variable == "rmse" ~ scale_y_continuous(limits = rmse_lims),
  variable == "time" ~ scale_y_continuous(limits = time_lims)
)) + 
  ggh4x::facetted_pos_scales(x = list(
    radius_only == FALSE ~ scale_x_continuous(labels = scales::label_math(r == .x)),
    radius_only == TRUE ~ scale_x_continuous(labels = scales::label_math(k == .x))
  ))


## Sample-size extrapolation 
df <- read.csv(file.path(int_path, "sensitivity_analysis.csv"))
df <- df %>% 
  rowwise() %>% 
  mutate(hyperparameter = max(k, r, na.rm=TRUE)) %>% 
  filter(hyperparameter %in% c(0.125, 20)) %>% 
  mutate(loss = (estimate - truth)^2) %>%
  group_by(estimator, hyperparameter, n) %>%
  summarise(rmse = sqrt(mean(loss)), time = mean(inference_time)) %>%
  melt(measure.vars = c("rmse", "time"))

samplesize <- ggplot(df, aes(x = n, y = value, group = estimator, colour = estimator)) +
  geom_point() + 
  geom_line() + 
  geom_vline(data = data.frame(variable = "rmse", n = 250), aes(xintercept = n), linetype = "dashed") + 
  facet_grid(variable ~ 1, scales = "free", switch = "y") +
  scale_estimator(df) +
  theme_bw(base_size = text_size) +
  labs(x = "Sample size (n)", colour = "Neighbourhood") + 
  ggh4x::facetted_pos_scales(y = list(
    variable == "rmse" ~ scale_y_continuous(limits = rmse_lims),
    variable == "time" ~ scale_y_continuous(limits = time_lims)
  )) + 
 guides(colour = guide_legend(byrow = TRUE)) + 
 theme(
  axis.title.y = element_blank(),
  axis.text.y = element_blank(),
  axis.ticks.y = element_blank(),
  strip.text.y = element_blank(),
  strip.text.x = element_blank(),
  strip.background = element_blank(),
  strip.placement = "outside", 
  legend.text.align = 0,                  # left justify legend
  legend.key.height = unit(2.25, "lines")        # increase legend key height
) 
  
figure <- egg::ggarrange(hyperparams, samplesize, nrow = 1, widths = c(2, 1))

ggsv(figure, file = "neighbourhood", width = 11.5, height = 5, path = img_path)


