source(file.path("src", "plotting.R"))
img_path <- file.path("img", "supplement")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)
radiuslabel <- "Disc radius"
depthlabel  <- "Propagation layers (depth)"
widthlabel  <- "Propagation channels (width)"

# ---- Load results: Disc radius vs depth ----

int_path <- file.path("intermediates", "supplement", "discradius")

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

df_radius <- df

# ---- Load results: Width vs. depth ----

int_path <- file.path("intermediates", "supplement", "factorialexperiment")

# Load the loss function per epoch files:
all_dirs  <- list.dirs(path = int_path, recursive = TRUE)
runs_dirs <- all_dirs[which(grepl("runs_", all_dirs))]
depth <- 1:6
width <- 2^(2:8)
risk_list <- expand.grid(depth, width) %>% 
  apply(1, function(study) {
    depth <- study[1]
    width <- study[2]
    path <- file.path(int_path, paste0("runs_GNN_", "depth", depth, "_width", width))
    loss_per_epoch <- read.csv(file.path(path, "loss_per_epoch.csv"), header = FALSE)
    colnames(loss_per_epoch) <- c("training", "validation")
    loss_per_epoch$epoch <- 0:(nrow(loss_per_epoch) - 1)
    loss_per_epoch$depth <- depth
    loss_per_epoch$width <- width
    loss_per_epoch <- loss_per_epoch[which.min(loss_per_epoch$validation), ]
    df <- loss_per_epoch
    
    df$training_time  <- as.numeric(read.csv(file.path(path, "train_time.csv"), header = FALSE)) 
    df$training_time  <- df$training_time/3600 
    
    df$inference_time  <- as.numeric(read.csv(file.path(path, "inference_time.csv"), header = TRUE)) 
    
    return(df)
  })

df <- do.call("rbind", risk_list)
nrow(df) == length(depth) * length(width) 

df$width <- as.ordered(df$width)
df$depth <- as.ordered(df$depth)

df_width <- df

# ---- Plotting ----

risk_scale <- range(c(df_radius$validation, df_width$validation))
time_scale <- range(c(df_radius$inference_time, df_width$inference_time))
traintime_scale <- range(c(df_radius$training_time, df_width$training_time))

risk1 <- ggplot(df_width) +
  geom_tile(aes(x = depth, y = width, fill = validation)) + 
  scale_fill_viridis_c(option = "magma", limits = risk_scale) + 
  theme_bw() +
  labs(fill = "Bayes risk", x = depthlabel, y = widthlabel) + 
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) + 
  theme(legend.position = "top")

risk2 <- ggplot(df_radius) +
  geom_tile(aes(x = depth, y = radius, fill = validation)) + 
  scale_fill_viridis_c(option = "magma", limits = risk_scale) + 
  theme_bw() +
  labs(fill = "Bayes risk", x = depthlabel, y = radiuslabel) + 
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) + 
  theme(legend.position = "top")

traintime1 <- ggplot(df_width) +
 geom_tile(aes(x = depth, y = width, fill = training_time)) +
 scale_fill_viridis_c(option = "magma", limits = traintime_scale) +
 theme_bw() +
 labs(fill = "Training time (hrs)", x = depthlabel, y = widthlabel) +
 scale_x_discrete(expand = c(0, 0)) +
 scale_y_discrete(expand = c(0, 0)) +
 theme(legend.position = "top")

traintime2 <- ggplot(df_radius) +
 geom_tile(aes(x = depth, y = radius, fill = training_time)) +
 scale_fill_viridis_c(option = "magma", limits = traintime_scale) +
 theme_bw() +
 labs(fill = "Training time (hrs)", x = depthlabel, y = radiuslabel) +
 scale_x_discrete(expand = c(0, 0)) +
 scale_y_discrete(expand = c(0, 0)) +
 theme(legend.position = "top")

time1 <- ggplot(df_width) +
  geom_tile(aes(x = depth, y = width, fill = inference_time)) + 
  scale_fill_viridis_c(option = "magma", limits = time_scale) + 
  theme_bw() +
  labs(fill = "Inference time (s)", x = depthlabel, y = widthlabel) + 
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) + 
  theme(legend.position = "top") 

time2 <- ggplot(df_radius) +
  geom_tile(aes(x = depth, y = radius, fill = inference_time)) + 
  scale_fill_viridis_c(option = "magma", n.breaks = 3, limits = time_scale) + 
  theme_bw() +
  labs(fill = "Inference time (s)", x = depthlabel, y = radiuslabel) + 
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) + 
  theme(legend.position = "top") 

# Combine into a single plot
risk1 <- risk1 + theme(axis.title.x = element_blank()) 
risk2 <- risk2 + theme(legend.position = "none")
traintime1 <- traintime1 + theme(axis.title.y = element_blank(), axis.title.x = element_blank())
traintime2 <- traintime2 + theme(axis.title.y = element_blank(), legend.position = "none")
time1 <- time1 + theme(axis.title.y = element_blank(), axis.title.x = element_blank()) 
time2 <- time2 + theme(axis.title.y = element_blank(), legend.position = "none") 
figure <- egg::ggarrange(risk1, traintime1, time1, risk2, traintime2, time2, nrow = 2)

ggsv(figure, file = "radius_and_width_vs_depth", width = 9.5, height = 6.7, path = img_path)


# ---- Sensitivity analysis of the radius with respect to n ----

int_path <- file.path("intermediates", "supplement", "neighbours")
df <- read.csv(file.path(int_path, "estimates.csv"))

## RMSE
df <- df %>%
  mutate(loss = (estimate - truth)^2) %>%
  group_by(estimator, n) %>% 
  # group_by(parameter, .add = TRUE) %>% # group by parameter to illustrate that it's not just range parameters that are affected
  summarise(rmse = sqrt(mean(loss)))

breaks <- unique(df$n)
breaks <- breaks[breaks != 60]

text_size <- 14

rmse1 <- ggplot(data = df, aes(x = n, y = rmse, colour = estimator, group = estimator, linetype = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  # facet_wrap(~parameter, labeller = param_labeller) +
  labs(colour = "", linetype = "", x = expression(n), y = "RMSE") +
  scale_estimator(df) +
  scale_estimator(df, scale = "linetype", values = estimator_linetypes) +
  theme_bw(base_size = text_size) +
  theme(
    strip.text.x = element_text(size = 12),
    legend.text.align = 0,
    panel.grid = element_blank(),
    strip.background = element_blank()
  )
rmse1

## Inference time
df <- read.csv(file.path(int_path, "runtime_singledataset.csv"))

time1 <- ggplot(data = df, aes(x = n, y = time, colour = estimator, group = estimator, linetype = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  labs(
    colour = "Neighbourhood definition", 
    x = expression(n), 
    y = "Inference time (s)"
  ) +
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

rmse1 <- rmse1 + theme(legend.position = "none")


int_path <- file.path("intermediates", "supplement", "discradius")
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

rmse2 <- ggplot(data = df, aes(x = n, y = rmse, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  labs(colour = "", x = expression(n), y = "RMSE") +
  theme_bw(base_size = text_size) +
  theme(
    legend.text.align = 0,
    panel.grid = element_blank(),
    strip.background = element_blank()
  )

time2 <- ggplot(data = df,
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

rmse2 <- rmse2 + theme(legend.position = "none")
figure <- egg::ggarrange(rmse1, time1, rmse2, time2, nrow = 2)

ggsv(figure, file = "neighbourhood_radius_vs_n", width = 10.9, height = 6.6, path = img_path)
