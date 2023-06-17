library("optparse")
option_list <- list(
  make_option("--model", type="character", default=NULL, metavar="character")
)
opt_parser  <- OptionParser(option_list=option_list)
model       <- parse_args(opt_parser)$model

source("src/plotting.R")

int_path <- paste0("intermediates/experiments/architectures/", model)
img_path <- paste0("img/experiments/architectures/", model)
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)


# ---- Risk function ----

# Load in estimates + true parameters
df <- read.csv(paste0(int_path, "/estimates_test.csv"))

df %>%
  mutate(loss = loss(estimate, truth)) %>%
  group_by(estimator) %>%
  summarise(risk = mean(loss), sd = sd(loss)/sqrt(length(loss))) %>%
  write.csv(file = paste0(img_path, "/risk.csv"), row.names = F)


# ---- Sampling distributions ----

# Load in estimates + true parameters
df <- read.csv(paste0(int_path, "/estimates_scenarios.csv"))

snk <- lapply(unique(df$k), function(K) {
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

  0
})
