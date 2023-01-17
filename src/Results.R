library("optparse")
option_list <- list(
  make_option("--model", type="character", default=NULL,
              help="The statistical model: for example, 'GaussianProcess/nuFixed'", 
              metavar="character")
)
opt_parser  <- OptionParser(option_list=option_list)
model       <- parse_args(opt_parser)$model

source("src/plotting.R")

parameter_labels <- c(
  "σ"  = expression(sigma),
  "ρ"  = expression(rho)
)

splitestimator <- function(df) {
  # df <- separate(data = df, col = estimator, into = c("globalpool", "propagationwidth"), sep = "_")
  # df$propagationwidth <- gsub("nh",  "", df$propagationwidth)
  # df$propagationwidth <-as.numeric(df$propagationwidth)
  # return(df)
  
  df %>% 
    separate(col = estimator, into = c("globalpool", "propagationwidth"), sep = "_") %>% 
    mutate(propagationwidth = gsub("nh",  "", propagationwidth)) %>% 
    mutate(propagationwidth = as.numeric(propagationwidth)) 
  

}

# ----------------------
# ---- Experiment 0 ----
# ----------------------

int_path <- paste0("intermediates/GNN/", model, "/experiment0")
img_path <- paste0("img/GNN/", model, "/experiment0")
results_path <- paste0("results/GNN/", model, "/experiment0")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)
dir.create(results_path, recursive = TRUE, showWarnings = FALSE)


# ---- Risk function ----

# Load in estimates + true parameters
df <- read.csv(paste0(int_path, "/estimates.csv"))
df <- splitestimator(df)

df <- df %>%
  group_by(globalpool, propagationwidth, trial) %>% 
  summarise(risk = MAE(estimate, truth)) %>%
  summarise(avrisk = mean(risk), sdrisk = sd(risk)) 

df %>% write.csv(file = paste0(results_path, "/risk.csv"), row.names = F)

gg <- ggplot(df) + 
  geom_line(aes(x = propagationwidth, y = avrisk, colour = globalpool)) + 
  labs(x = "Number of channels in propagation module", 
       y = risklabel, 
       colour = "Global pooling\nmodule") + 
  theme_bw()

ggsave(gg, file = "risk.pdf", width = 6, height = 4, device = "pdf", path = img_path)

# ---- Test time ----

df <- read.csv(paste0(int_path, "/runtime.csv"))
df$trial <- rep(1:(nrow(df)/10), each = 10) # TODO delete this after rerunning the julia code 
df <- df[df$trial != 1, ] # remove the first trial, which is subject to noise due to code compilation time
df <- splitestimator(df)
df <- df %>% 
  group_by(globalpool, propagationwidth) %>% 
  summarise(avtime = mean(time), sdtime = sd(time))

gg <- ggplot(df) + 
  geom_line(aes(x = propagationwidth, y = avtime, colour = globalpool)) + 
  labs(x = "Number of channels in propagation module", 
       y = "Computational time (s)", 
       colour = "Global pooling\nmodule") + 
  theme_bw()

ggsave(gg, file = "time.pdf", width = 6, height = 4, device = "pdf", path = img_path)


# ----------------------
# ---- Experiment 1 ----
# ----------------------

int_path <- paste0("intermediates/GNN/", model)
img_path <- paste0("img/GNN/", model, "/experiment1")
results_path <- paste0("results/GNN/", model, "/experiment1")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)
dir.create(results_path, recursive = TRUE, showWarnings = FALSE)


# ---- Risk function ----

# Load in estimates + true parameters
df <- read.csv(paste0(int_path, "/estimates_test.csv"))

df %>%
  group_by(estimator, trial) %>%
  summarise(risk = MAE(estimate, truth)) %>%
  summarise(risk_average = mean(risk), risk_sd = sd(risk)) %>%
  write.csv(file = paste0(results_path, "/risk.csv"), row.names = F)


# ---- Joint distribution of the estimators ----

# Load in estimates + true parameters
df <- read.csv(paste0(int_path, "/estimates_scenarios.csv"))


lapply(unique(df$k), function(K) {
  ggsave(
    plotdistribution(df %>% filter(k == K), parameter_labels = parameter_labels),
    file = paste0("boxplot", K, ".pdf"),
    width = 8, height = 4, device = "pdf", path = img_path
  )

  ggsave(
    plotdistribution(df %>% filter(k == K), parameter_labels = parameter_labels, type = "scatter")[[1]],
    file = paste0("scatterplot", K, ".pdf"),
    width = 5, height = 5, device = "pdf", path = img_path
  )
})


# ----------------------
# ---- Experiment 2 ----
# ----------------------

int_path <- paste0("intermediates/GNN/", model)
img_path <- paste0("img/GNN/", model, "/experiment2")
results_path <- paste0("results/GNN/", model, "/experiment2")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)
dir.create(results_path, recursive = TRUE, showWarnings = FALSE)

boldtheta <- bquote(bold(theta))
estimator_labels <- c(
  "GNN" = bquote(hat(.(boldtheta))[GNN]("·") * " : " * tilde(S)),
  "GNN_S" = bquote(hat(.(boldtheta))[GNN]("·") * " : " * S),
  "GNN_Svariable" = bquote(hat(.(boldtheta))[GNN]("·") * " : " * S[k] * ", k = 1, ..., K"),
  "WGNN" = bquote(hat(.(boldtheta))[WGNN]("·") * " : " * tilde(S)),
  "WGNN_S" = bquote(hat(.(boldtheta))[WGNN]("·") * " : " * S),
  "WGNN_Svariable" = bquote(hat(.(boldtheta))[WGNN]("·") * " : " * S[k] * ", k = 1, ..., K")
)



# ---- Risk function ----

# Load in estimates + true parameters
df <- read.csv(paste0(int_path, "/estimates_test_S.csv"))

df %>%
  group_by(estimator, trial) %>%
  summarise(risk = MAE(estimate, truth)) %>%
  summarise(risk_average = mean(risk), risk_sd = sd(risk)) %>%
  write.csv(file = paste0(results_path, "/risk.csv"), row.names = F)


# ---- Joint distribution of the estimators ----

# Load in estimates + true parameters
df <- read.csv(paste0(int_path, "/estimates_scenarios_S.csv"))


lapply(unique(df$k), function(K) {
  ggsave(
    plotdistribution(df %>% filter(k == K), 
                     parameter_labels = parameter_labels, 
                     estimator_labels = estimator_labels),
    file = paste0("boxplot", K, ".pdf"),
    width = 8, height = 4, device = "pdf", path = img_path
  )

  ggsave(
    plotdistribution(df %>% filter(k == K), 
                     parameter_labels = parameter_labels, 
                     estimator_labels = estimator_labels, 
                     type = "scatter")[[1]],
    file = paste0("scatterplot", K, ".pdf"),
    width = 5, height = 5, device = "pdf", path = img_path
  )
})
