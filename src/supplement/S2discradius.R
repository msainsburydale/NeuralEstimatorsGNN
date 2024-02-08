int_path <- file.path("intermediates", "supplement", "discradius")
img_path <- file.path("img", "supplement", "discradius")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

source(file.path("src", "plotting.R"))

# Load the loss function per epoch files:
all_dirs  <- list.dirs(path = int_path, recursive = TRUE)
runs_dirs <- all_dirs[which(grepl("runs_", all_dirs))]
depth <- 1:6
radius <- c(0.05, 0.1, 0.15, 0.2, 0.25, 0.3) 
risk_list <- expand.grid(depth, radius) %>% 
  apply(1, function(study) {
    depth <- study[1]
    radius <- study[2]
    path <- file.path(int_path, paste0("runs_GNN_", "depth", depth, "_radius", radius))
    loss_per_epoch <- read.csv(file.path(path, "loss_per_epoch.csv"), header = FALSE)
    colnames(loss_per_epoch) <- c("training", "validation")
    loss_per_epoch$epoch <- 0:(nrow(loss_per_epoch) - 1)
    loss_per_epoch$depth <- depth
    loss_per_epoch$radius <- radius
    loss_per_epoch <- loss_per_epoch[which.min(loss_per_epoch$validation), ]
    df <- loss_per_epoch
    
    df$training_time  <- as.numeric(read.csv(file.path(path, "train_time.csv"), header = FALSE)) 
    df$training_time  <- df$training_time/3600 
  
    df$inference_time  <- as.numeric(read.csv(file.path(path, "inference_time.csv"), header = TRUE)) 
    
    return(df)
  })

df <- do.call("rbind", risk_list)
nrow(df) == length(depth) * length(radius) 

df$radius <- as.ordered(df$radius)
df$depth <- as.ordered(df$depth)

radiuslabel <- "Disc radius"
depthlabel <- "Propagation layers (depth)"

risk_plot <- ggplot(df) +
  geom_tile(aes(x = depth, y = radius, fill = validation)) + 
  scale_fill_viridis_c(option = "magma") + 
  theme_bw() +
  labs(fill = "Bayes risk", x = depthlabel, y = radiuslabel) + 
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) + 
  theme(legend.position = "top")

traintime_plot <- ggplot(df) +
  geom_tile(aes(x = depth, y = radius, fill = training_time)) + 
  scale_fill_viridis_c(option = "magma") + 
  theme_bw() +
  labs(fill = "Training time (hrs)", x = depthlabel, y = radiuslabel) + 
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) + 
  theme(legend.position = "top")

inferencetime_plot <- ggplot(df) +
  geom_tile(aes(x = depth, y = radius, fill = inference_time)) + 
  scale_fill_viridis_c(option = "magma", breaks = c(0.00075, 0.00125)) + 
  theme_bw() +
  labs(fill = "Inference time (s)", x = depthlabel, y = radiuslabel) + 
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) + 
  theme(legend.position = "top") 

# Combine into a single plot
risk_plot <- risk_plot # + theme(axis.title.x = element_blank()) 
traintime_plot <- traintime_plot + theme(axis.title.y = element_blank()) 
inferencetime_plot <- inferencetime_plot + theme(axis.title.y = element_blank()) # + theme(axis.title.x = element_blank()) 
figure <- egg::ggarrange(risk_plot, traintime_plot, inferencetime_plot, nrow = 1)

ggsave(
  figure,
  file = "radius_vs_depth.pdf",
  width = 9.5, height = 3.7, path = img_path, device = "pdf"
)

# ---- Sensitivity analysis of the radius with respect to n ----

df <- read.csv(file.path(int_path, "radius_vs_n.csv"))

## RMSE
df <- df %>%
  mutate(loss = (estimate - truth)^2) %>%
  group_by(radius, n) %>% 
  # group_by(parameter, .add = TRUE) %>% # group by parameter to illustrate that it's not just range parameters that are affected
  summarise(rmse = sqrt(mean(loss)), time = mean(inference_time))

df$estimator <- ordered(df$radius)

breaks <- unique(df$n)
breaks <- breaks[breaks != 60]

text_size <- 14

figure_rmse <- ggplot(data = df, aes(x = n, y = rmse, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  labs(colour = "", x = expression(n), y = "RMSE") +
  theme_bw(base_size = text_size) +
  theme(
    strip.text.x = element_text(size = 12),
    legend.text.align = 0,
    panel.grid = element_blank(),
    strip.background = element_blank()
  )

figure_time <- ggplot(data = df,
                      aes(x = n, y = time, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  labs(
    colour = "Disc radius", 
    x = expression(n), 
    y = "Inference time (s)"
  ) +
  theme_bw(base_size = text_size) +
  guides(colour = guide_legend(byrow = TRUE)) +
  theme(
    legend.text.align = 0,
    panel.grid = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_blank(), 
    legend.spacing.y = unit(1, "lines")
  )

figure <- egg::ggarrange(figure_rmse + theme(legend.position = "none"), figure_time, nrow = 1)

ggsave(
  figure,
  file = "rmse_runtime_vs_n.pdf",
  width = 12, height = 4, path = img_path, device = "pdf"
)