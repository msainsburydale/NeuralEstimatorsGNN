library("optparse")
option_list <- list(
  make_option("--model", type="character", default=NULL, metavar="character")
)
opt_parser  <- OptionParser(option_list=option_list)
model       <- parse_args(opt_parser)$model

int_path <- paste("intermediates/experiments/samplesize", model, sep = "/")
img_path <- paste("img/experiments/samplesize", model, sep = "/")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

source("src/plotting.R")


# ---- Risk function ----

df1 <- read.csv(paste0(int_path, "/radius/estimates_test.csv"))
df2 <- read.csv(paste0(int_path, "/fixednum/estimates_test.csv"))

df1$neighbours <- "radius"
df2$neighbours <- "fixednum"

df <- rbind(df1, df2)

df <- df %>% 
  # filter(estimator %in% c("GNN3", "MAP")) %>% 
  filter(estimator %in% c("GNN3")) %>% 
  mutate(estimator = paste(estimator, neighbours))

## Collapse MAP into a single estimator
# df$estimator <- gsub("MAP radius", "MAP", df$estimator)
# df$estimator <- gsub("MAP fixednum", "MAP", df$estimator)

## Bayes risk with respect to absolute error
df <- df %>%
  mutate(loss = abs(estimate - truth)) %>% 
  group_by(estimator, parameter, n) %>% 
  summarise(risk = mean(loss), sd = sd(loss)/sqrt(length(loss)))

breaks <- unique(df$n)
breaks <- breaks[breaks != 60]

param_labeller <- label_parsed
df <- mutate_at(df, .vars = "parameter", .funs = factor, levels = names(parameter_labels), labels = parameter_labels)

figure <- ggplot(data = df,
                 aes(x = n, y = risk, colour = estimator, group = estimator)) +
  geom_point() +
  geom_line(alpha = 0.75) +
  facet_wrap(~parameter, labeller = param_labeller) + 
  labs(colour = "", 
       x = expression(n), 
       y = expression(r[Omega](hat(theta)("·")))) +
  scale_x_continuous(breaks = breaks) +
  scale_estimator(df) +
  theme_bw() +
  theme(
    legend.text=element_text(size = 12),
    text = element_text(size = 12),
    strip.text.x = element_text(size = 12),
    legend.text.align = 0,
    panel.grid = element_blank(),
    strip.background = element_blank()
  )

ggsave(
  figure,
  file = "risk_vs_n_neighbour-definition.pdf",
  width = 12, height = 4, path = img_path, device = "pdf"
)

