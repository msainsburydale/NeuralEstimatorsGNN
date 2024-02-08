int_path <- file.path("intermediates", "supplement", "factorialexperiment")
img_path <- file.path("img", "supplement", "factorialexperiment")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

source(file.path("src", "plotting.R"))

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

widthlabel <- "Propagation channels (width)"
depthlabel <- "Propagation layers (depth)"

risk_plot <- ggplot(df) +
  geom_tile(aes(x = depth, y = width, fill = validation)) + 
  scale_fill_viridis_c(option = "magma") + 
  theme_bw() +
  labs(fill = "Bayes risk", x = depthlabel, y = widthlabel) + 
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) + 
  theme(legend.position = "top")

traintime_plot <- ggplot(df) +
  geom_tile(aes(x = depth, y = width, fill = training_time)) + 
  scale_fill_viridis_c(option = "magma") + 
  theme_bw() +
  labs(fill = "Training time (hrs)", x = depthlabel, y = widthlabel) + 
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) + 
  theme(legend.position = "top")

inferencetime_plot <- ggplot(df) +
  geom_tile(aes(x = depth, y = width, fill = inference_time)) + 
  scale_fill_viridis_c(option = "magma", breaks = c(0.002, 0.004)) + 
  theme_bw() +
  labs(fill = "Inference time (s)", x = depthlabel, y = widthlabel) + 
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) + 
  theme(legend.position = "top") 

# Combine into a single plot
risk_plot <- risk_plot + theme(axis.title.x = element_blank()) 
traintime_plot <- traintime_plot + theme(axis.title.y = element_blank()) 
inferencetime_plot <- inferencetime_plot + theme(axis.title.y = element_blank(), axis.title.x = element_blank()) 
figure <- egg::ggarrange(risk_plot, traintime_plot, inferencetime_plot, nrow = 1)

ggsv(figure, file = "depth_vs_width", width = 9.5, height = 3.7, path = img_path)
