suppressMessages({
library("spatstat")
library("ggplot2")
library("latex2exp")
library("dplyr")  
options(dplyr.summarise.inform = FALSE) # Suppress summarise info
})
source("src/plotting.R")

set.seed(1)

# vary the parameters
n <- 250
lambda <- c(10, 25, 50, 90)
mu <- round(n / lambda, 2)
r <- 0.1

# Now simulate spatial point process for every combination of (lambda, mu)
S_list <- lapply(1:length(lambda), function(i) {
  pts <- rMatClust(kappa = lambda[i], r = r, mu = mu[i])
  data.frame(x=pts$x, y = pts$y, lambda = lambda[i], r = r, mu = mu[i])
})
df <- do.call("rbind", S_list) 

df$facet_var <- paste0("$\\lambda$ = ", df$lambda, ", $\\mu$ = ", df$mu)
df$facet_var <- as.factor(df$facet_var)
levels(df$facet_var) <- sapply(levels(df$facet_var), TeX)

df$r<- factor(df$r)
levels(df$r) <- sapply(paste("$r$ = ", levels(df$r)), TeX)

figure <- ggplot(df) + 
  geom_point(aes(x = x, y = y)) + 
  facet_grid(~facet_var, labeller = label_parsed) +
  labs(x = expression(s[1]), y = expression(s[2])) + 
  scale_x_continuous(breaks = c(0.2, 0.5, 0.8)) + 
  scale_y_continuous(breaks = c(0.2, 0.5, 0.8)) + 
  coord_fixed() + 
  theme_bw()

ggsv(figure, file = "spatialpatterns", width = 7.3, height = 2.4, path = "img")


# ---- Scaling spatial locations (relevant for applications) ----

S <- rMatClust(kappa = 10, r = r, mu = 10, win = owin(xrange=c(0,2), yrange=c(0,0.8)))
r <- max(max(S$x) - min(S$x), max(S$y) - min(S$y))
df1 <- data.frame(x = S$x, y = S$y)
df2 <- df1/r
df1$set <- "Original locations"
df2$set <- "Scaled locations"
df <- rbind(df1, df2)
ggplot(df) + 
  geom_point(aes(x, y)) + 
  facet_wrap(vars(set)) + 
  labs(x = expression(s[1]), y = expression(s[2])) + 
  coord_fixed() + 
  theme_bw()



