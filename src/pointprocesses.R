suppressMessages({
library("spatstat")
library("ggplot2")
library("latex2exp")
library("dplyr")  
options(dplyr.summarise.inform = FALSE) # Suppress summarise info
})


set.seed(1)

# vary the parameters
n <- 250
kappa <- c(10, 25, 50, 90)
mu <- round(n / kappa, 2)
r <- c(0.1, 0.2, 0.3)

# Now simulate spatial point process for every combination of (kappa, mu) and r
l <- lapply(1:length(kappa), function(i) {
  lapply(1:length(r), function(j) {
    pts <- rMatClust(kappa = kappa[i], r = r[j], mu = mu[i])
    data.frame(x=pts$x, y = pts$y, kappa = kappa[i], r = r[j], mu = mu[i])
  })
})
l <- unlist(l,recursive=F)
df <- do.call("rbind", l)

df$facet_var <- paste0("$\\lambda$ = ", df$kappa, ", $\\mu$ = ", df$mu)
df$facet_var <- as.factor(df$facet_var)
levels(df$facet_var) <- sapply(levels(df$facet_var), TeX)

df$r<- factor(df$r)
levels(df$r) <- sapply(paste("$r$ = ", levels(df$r)), TeX)

figure <- ggplot(df %>% filter(r == 'r * " =  0.1"')) + 
  geom_point(aes(x = x, y = y)) + 
  # facet_grid(r~facet_var, labeller = label_parsed) +
  facet_grid(~facet_var, labeller = label_parsed) +
  labs(x = expression(s[1]), y = expression(s[2])) + 
  scale_x_continuous(breaks = c(0.2, 0.5, 0.8)) + 
  scale_y_continuous(breaks = c(0.2, 0.5, 0.8)) + 
  coord_fixed() + 
  theme_bw()

ggsave(figure, file = "spatialpatterns.pdf", width = 7.3, height = 2.4, path = "img", device = "pdf")


# # histogram of spatial distances
# hdf <- df %>% 
#   filter(r == 'r * " =  0.1"') %>%
#   group_by(facet_var) %>%
#   summarise(h = c(dist(matrix(c(x, y), ncol = 2, byrow = F)))) #%>% 
#   # filter(h < 0.15) # only consider points within a distance of hmax
# 
# figure <- ggplot(hdf) +
#   geom_histogram(aes(x = h), bins = 30) +
#   facet_grid(~facet_var, labeller = label_parsed) + 
#   labs() + 
#   theme_bw()
# 
# ggsave(figure, file = "spatialpatternsdistances.pdf", width = 7.3, height = 2.7, path = "img", device = "pdf")
# 



