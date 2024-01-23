int_path <- file.path("intermediates", "supplement", "neighbours")
img_path <- paste("img", "supplement", "neighbours")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

# ---- Risk function ----

df <- read.csv(file.path(int_path, "/estimates.csv"))

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

## TODO plot the inference time as well

ggsave(
  figure,
  file = "risk_vs_n_neighbour-definition.pdf",
  width = 12, height = 4, path = img_path, device = "pdf"
)
