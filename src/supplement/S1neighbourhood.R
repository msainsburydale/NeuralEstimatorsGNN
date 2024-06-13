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
  #scale_estimator(df) +
  theme_bw(base_size = text_size) +
  theme(
    strip.text.x = element_text(size = 12),
    legend.text.align = 0,
    strip.background = element_blank()
  )

ggsv(figure_training, file = "risk_vs_epoch", width = 10.6, height = 4.5, path = img_path)


# ---- Sensitivity analysis: as a function of hyperparameter ----

#TODO clean code (can delete)

## Results data frame
df <- read.csv(file.path(int_path, "sensitivity_analysis.csv"))
df <- df %>% 
  filter(n == 250, is.na(k) | k <= 20) %>%
  mutate(loss = (estimate - truth)^2) %>%
  group_by(k, r, estimator, n) %>%
  summarise(rmse = sqrt(mean(loss)), time = mean(inference_time)) %>% 
  rowwise() %>% 
  mutate(hyperparameter = max(k, r, na.rm=TRUE))

# estimator_names <- c(
#   "fixedradius" = "Disc of fixed radius r", 
#   "fixedradiusmaxk" = "Subset of k neighbours\nwithin a disc of fixed radius r=0.1", 
#   "knearest" = "k-nearest neighbours", 
#   "maxmin" = "k-nearest neighbours\nsubject to a maxmin ordering"
# )

p_rmse <- ggplot(df, aes(x = hyperparameter, y = rmse)) +
  geom_point() + 
  geom_line() + 
  ylim(c(0.03, 0.1)) + 
  labs(x = "Hyperparameter (r or k)", y = "RMSE") + 
  facet_wrap(vars(estimator), scales = "free_x", labeller = as_labeller(estimator_names), nrow = 1) +
  theme_bw() +
  theme(
    axis.text.y = element_text(size = 11),
    axis.title.y = element_text(size = 12),
    axis.text.x = element_blank(), 
    axis.ticks.x = element_blank(), 
    axis.title.x = element_blank(), 
    strip.text.x = element_text(size = 11)
  )

p_time <- ggplot(df, aes(x = hyperparameter, y = time)) +
  geom_point() + 
  geom_line() + 
  ylim(c(0, 0.005)) + 
  labs(x = "Hyperparameter", y = "Inference time (s)") + 
  facet_wrap(vars(estimator), scales = "free_x", labeller = as_labeller(estimator_names), nrow = 1) + 
  theme_bw() + 
  theme(        
    strip.background = element_blank(),
    strip.text.x = element_blank(), 
    axis.title.y = element_text(size = 12),
    axis.text.y = element_text(size = 11),
    axis.title.x = element_text(size = 12),
    axis.text.x = element_text(size = 11)
    )

# Edit axis tick labels
breaks <- c(0.04, 0.08, 0.12, 5, 10, 15, 20)
label_breaks <- function(breaks) sapply(breaks, function(x) if(x < 1) str_glue("r = {x}") else str_glue("k = {x}"))
labels <- label_breaks(breaks)
names(labels) <- as.character(breaks)
label_breaks2 <- function(breaks) labels[as.character(breaks)]
p_time <- p_time + scale_x_continuous(labels = label_breaks2)

figure <- egg::ggarrange(p_rmse, p_time, nrow = 2)

ggsv(figure, file = "neighbourhood", width = 11.5, height = 5, path = img_path)


# ---- Sample-size extrapolation ----


## Results data frame
df <- read.csv(file.path(int_path, "sensitivity_analysis.csv"))
df <- df %>% 
  rowwise() %>% 
  mutate(hyperparameter = max(k, r, na.rm=TRUE)) %>% 
  filter(hyperparameter %in% c(0.1, 20)) %>% 
  mutate(loss = (estimate - truth)^2) %>%
  group_by(estimator, hyperparameter, n) %>%
  summarise(rmse = sqrt(mean(loss)), time = mean(inference_time))

ggplot(df, aes(x = n, y = rmse, group = estimator, linetype = estimator)) + 
  geom_point() + 
  geom_line() + 
  lims(y = c(0.03, 0.1)) + 
  theme_bw()




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

rmse_lims <- c(0.03, 0.1)
time_lims <- c(0.001, 0.005)

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
  filter(hyperparameter %in% c(0.1, 20)) %>% #TODO change 0.1 to 0.15 after rerunning 
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
  guides(colour = guide_legend(byrow = TRUE)) + # needed to increase separation between legend entries
  theme(
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    strip.text.y = element_blank(),
    strip.text.x = element_blank(),
    strip.background = element_blank(),
    strip.placement = "outside", 
    legend.text.align = 0,                  # left justify legend
    legend.spacing.y = unit(0.75, "lines")   # increase separation between legend entries
  ) + 
  ggh4x::facetted_pos_scales(y = list(
    variable == "rmse" ~ scale_y_continuous(limits = rmse_lims),
    variable == "time" ~ scale_y_continuous(limits = time_lims)
  )) #+ 
  #scale_x_continuous(labels = scales::label_math(n == .x))
  
figure <- egg::ggarrange(hyperparams, samplesize, nrow = 1, widths = c(2, 1))

ggsv(figure, file = "neighbourhood", width = 11.5, height = 5, path = img_path)


# ---- Original ----

# 
# df$estimator[df$estimator == "combined"] <- "zcombined" # hack to make "combined" the final column
# estimator_names <- c(
#   "knearest" = "k-nearest neighbours", 
#   "maxmin" = "k-nearest neighbours subject \n to maxmin ordering", 
#   "fixedradius" = "Disc of fixed radius r\n", 
#   "zcombined" = "Union of k-nearest neighbours and \n k-nearest neighbours subject to maxmin ordering"
# )
# 
# r_rmse <- ggplot(data = df %>% filter(estimator == "fixedradius"), 
#                  aes(x = n, y = rmse, colour = r, group = r)) +
#   geom_point() +
#   facet_wrap(vars(estimator), labeller = as_labeller(estimator_names)) +
#   lims(y = rmse_lims) + 
#   geom_line(alpha = 0.75) +
#   labs(colour = "Disc radius r", x = expression(n), y = "RMSE") +
#   scale_color_brewer(palette=palette) +
#   theme_bw(base_size = text_size) +
#   theme(legend.position = "top", 
#         axis.text.x = element_blank(), 
#         axis.ticks.x = element_blank(), 
#         axis.title.x = element_blank()
#   )
# 
# 
# k_rmse <- ggplot(data = df %>% filter(estimator != "fixedradius"), aes(x = n, y = rmse, colour = k, group = k)) +
#   geom_point() +
#   facet_wrap(vars(estimator), labeller = as_labeller(estimator_names)) +
#   lims(y = rmse_lims) + 
#   geom_line(alpha = 0.75) +
#   labs(colour = "Number of neighbours k", x = expression(n), y = "RMSE") +
#   scale_color_brewer(palette=palette) +
#   theme_bw(base_size = text_size) + 
#   theme(legend.position = "top", 
#         axis.text.y = element_blank(), 
#         axis.ticks.y = element_blank(), 
#         axis.title.y = element_blank(), 
#         axis.text.x = element_blank(), 
#         axis.ticks.x = element_blank(), 
#         axis.title.x = element_blank())
# 
# r_time <- ggplot(data = df %>% filter(estimator == "fixedradius"),
#                  aes(x = n, y = time, colour = r, group = r)) +
#   geom_point() +
#   lims(y = time_lims) + 
#   facet_wrap(vars(estimator), labeller = as_labeller(estimator_names)) +
#   geom_line(alpha = 0.75) +
#   labs(colour = "Number of neighbours, k", y = "Inference time (s)") +
#   theme_bw(base_size = text_size) +
#   scale_color_brewer(palette=palette) + 
#   theme(legend.position = "none", 
#         strip.background = element_blank(),
#         strip.text.x = element_blank()
#   )
# 
# k_time <- ggplot(data = df %>% filter(estimator != "fixedradius"),
#                  aes(x = n, y = time, colour = k, group = k)) +
#   geom_point() +
#   lims(y = time_lims) + 
#   facet_wrap(vars(estimator), labeller = as_labeller(estimator_names)) +
#   geom_line(alpha = 0.75) +
#   labs(colour = "Number of neighbours, k", y = "Inference time (s)") +
#   theme_bw(base_size = text_size) +
#   scale_color_brewer(palette=palette) + 
#   theme(legend.position = "none", 
#         axis.text.y = element_blank(), 
#         axis.ticks.y = element_blank(), 
#         axis.title.y = element_blank(), 
#         strip.background = element_blank(),
#         strip.text.x = element_blank())
# 
# figure <- egg::ggarrange(
#   r_rmse, k_rmse, 
#   r_time, k_time,
#   nrow = 2, 
#   widths = c(1, 3)
#   )
# 
# ggsv(figure, file = "neighbourhood", width = 12, height = 6, path = img_path)



# ---- Sensitivity analysis: as a function of n ----

# ## See the following link for a nice app demonstrating possible colour scales:
# ## https://colorbrewer2.org/#type=sequential&scheme=YlOrRd&n=5
# palette = "Reds"
# 
# ## Results data frame
# df <- read.csv(file.path(int_path, "sensitivity_analysis.csv"))
# df <- df %>% 
#   mutate(loss = (estimate - truth)^2) %>%
#   group_by(k, r, estimator, n) %>%
#   summarise(rmse = sqrt(mean(loss)), time = mean(inference_time))
# df$k <- ordered(df$k)
# df$r <- ordered(df$r)
# rmse_lims <- range(df$rmse)
# time_lims <- range(df$time)
# 
# df$estimator[df$estimator == "combined"] <- "zcombined" # hack to make "combined" the final column
# estimator_names <- c(
#   "knearest" = "k-nearest neighbours", 
#   "maxmin" = "k-nearest neighbours subject \n to maxmin ordering", 
#   "fixedradius" = "Disc of fixed radius r\n", 
#   "zcombined" = "Union of k-nearest neighbours and \n k-nearest neighbours subject to maxmin ordering"
# )
# 
# r_rmse <- ggplot(data = df %>% filter(estimator == "fixedradius"), 
#                  aes(x = n, y = rmse, colour = r, group = r)) +
#   geom_point() +
#   facet_wrap(vars(estimator), labeller = as_labeller(estimator_names)) +
#   lims(y = rmse_lims) + 
#   geom_line(alpha = 0.75) +
#   labs(colour = "Disc radius r", x = expression(n), y = "RMSE") +
#   scale_color_brewer(palette=palette) +
#   theme_bw(base_size = text_size) +
#   theme(legend.position = "top", 
#         axis.text.x = element_blank(), 
#         axis.ticks.x = element_blank(), 
#         axis.title.x = element_blank()
#   )
# 
# 
# k_rmse <- ggplot(data = df %>% filter(estimator != "fixedradius"), aes(x = n, y = rmse, colour = k, group = k)) +
#   geom_point() +
#   facet_wrap(vars(estimator), labeller = as_labeller(estimator_names)) +
#   lims(y = rmse_lims) + 
#   geom_line(alpha = 0.75) +
#   labs(colour = "Number of neighbours k", x = expression(n), y = "RMSE") +
#   scale_color_brewer(palette=palette) +
#   theme_bw(base_size = text_size) + 
#   theme(legend.position = "top", 
#         axis.text.y = element_blank(), 
#         axis.ticks.y = element_blank(), 
#         axis.title.y = element_blank(), 
#         axis.text.x = element_blank(), 
#         axis.ticks.x = element_blank(), 
#         axis.title.x = element_blank())
# 
# r_time <- ggplot(data = df %>% filter(estimator == "fixedradius"),
#                  aes(x = n, y = time, colour = r, group = r)) +
#   geom_point() +
#   lims(y = time_lims) + 
#   facet_wrap(vars(estimator), labeller = as_labeller(estimator_names)) +
#   geom_line(alpha = 0.75) +
#   labs(colour = "Number of neighbours, k", y = "Inference time (s)") +
#   theme_bw(base_size = text_size) +
#   scale_color_brewer(palette=palette) + 
#   theme(legend.position = "none", 
#         strip.background = element_blank(),
#         strip.text.x = element_blank()
#   )
# 
# k_time <- ggplot(data = df %>% filter(estimator != "fixedradius"),
#                  aes(x = n, y = time, colour = k, group = k)) +
#   geom_point() +
#   lims(y = time_lims) + 
#   facet_wrap(vars(estimator), labeller = as_labeller(estimator_names)) +
#   geom_line(alpha = 0.75) +
#   labs(colour = "Number of neighbours, k", y = "Inference time (s)") +
#   theme_bw(base_size = text_size) +
#   scale_color_brewer(palette=palette) + 
#   theme(legend.position = "none", 
#         axis.text.y = element_blank(), 
#         axis.ticks.y = element_blank(), 
#         axis.title.y = element_blank(), 
#         strip.background = element_blank(),
#         strip.text.x = element_blank())
# 
# figure <- egg::ggarrange(
#   r_rmse, k_rmse, 
#   r_time, k_time,
#   nrow = 2, 
#   widths = c(1, 3)
#   )
# 
# ggsv(figure, file = "neighbourhood", width = 12, height = 6, path = img_path)
