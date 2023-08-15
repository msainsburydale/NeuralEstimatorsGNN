library("NeuralEstimators")
library("JuliaConnectoR")
library("reshape2")
library("ggplot2")
library("dggrids")
library("dplyr")
library("FRK")
library("sp")
library("ggpubr")
library("spdep") # poly2nb

img_path <- "img/application/SST"
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

p = 3L # number of parameters

#TODO the scaling factor needs to account for the fact that we are including
#     neighbours: notice that without doing so, rho is estimated to be much larger using the neighbour approach.
#TODO add lower and upper bound plots in the supplementary material.
#TODO Timings

# ---- Helper functions ----

map_to_BAUs <- function(data_sp, sp_pols) {

  ## Suppress bindings warnings
  . <- BAU_name <- NULL

  ## Add BAU ID to the data frame of the SP object
  sp_pols$BAU_name <- as.character(row.names(sp_pols))

  ## Add coordinates to @data if not already there
  if(!(all(coordnames(sp_pols) %in% names(sp_pols@data))))
    sp_pols@data <- cbind(sp_pols@data,coordinates(sp_pols))

  ## Find which fields in the data object are not already declared in the BAUs
  diff_fields <- intersect(setdiff(names(data_sp),names(sp_pols)),names(data_sp))

  ## Create a data frame just of these fields
  data_df <- data_sp@data[diff_fields]

  ## Assign the CRS from sp_pols to data_sp. Note that the sp_pols
  ## are typically the BAUs object, and have not been altered
  ## significantly to this point (while data_sp has, and so
  ## its CRS is often NA).
  slot(data_sp, "proj4string") <- slot(sp_pols, "proj4string")
  data_over_sp <- FRK:::.parallel_over(data_sp, sp_pols)

  ## We now cbind the original data with data_over_sp
  data_over_sp <- cbind(data_df, data_over_sp)

  if(any(is.na(data_over_sp$BAU_name))) {  # data points at 180 boundary or outside BAUs -- remove
    ii <- which(is.na((data_over_sp$BAU_name)))
    data_sp <- data_sp[-ii,]
    data_over_sp <- data_over_sp[-ii,]
    warning("Removing data points that do not fall into any BAUs.")
  }

  new_sp_pts <- SpatialPointsDataFrame(
    coords=data_sp@coords,         # coordinates of summarised data
    data= data_over_sp ,                                # data frame
    proj4string = CRS(slot(data_sp, "proj4string")@projargs) # CRS of original data
    )

  new_sp_pts
}

# conversions from here: https://stackoverflow.com/a/1185413
xyz_conversion <- function(lonlat, R = 6371) {

  lon <- lonlat[1] * pi/180
  lat <- lonlat[2] * pi/180

  x = R * cos(lat) * cos(lon)
  y = R * cos(lat) * sin(lon)
  z = R * sin(lat)

  c(x, y, z)
}

chord_length <- function(S){
  S <- t(apply(S, 1, xyz_conversion))
  D <- dist(S, upper = T, diag = T)

  as.matrix(D)
}


# ---- Load the data ----

load("data/SST.rda")
df <- SST_sub_1000000
df$error <- df$bias  <- NULL # remove columns that will not be used
set.seed(1)
# df <- df[sample(1:nrow(df), 100000), ] # thin the data set for faster prototyping

## Remove impossible locations, and remove repetitions
df <- df %>%
  subset(lon < 180 & lon > -180 & lat < 90 & lat > -90) %>%
  distinct(lon, lat, .keep_all = TRUE)

## Detrend the data following Zammit-Mangion and Rougier (2020).
## Here, we define the residuals from a linear model, which will be used as
## the response variable for the analysis.
df$lat2 <- df$lat^2
df$Z    <- residuals(lm(sst ~ 1 + lat + lat2, data = df))
df$sst <- df$lat2 <- NULL

df_backup <- df

# ---- Bin the data ----

coordinates(df) = ~ lon + lat
slot(df, 'proj4string') <- CRS('+proj=longlat +ellps=sphere')
baus <- auto_BAUs(manifold = sphere(), type = "hex", isea3h_res = 5, data = df)

# relabel the BAUs to be from 1:N (by default, there are some missing numbers
# which can cause problems)
N <- length(baus)
for (i in 1:length(baus)) {
  baus@polygons[[i]]@ID <- as.character(i)
}
names(baus@polygons) <- 1:N
baus@data$id <- 1:N

x <- map_to_BAUs(df, sp_pols = baus)
x@data[, c("lon", "lat")] <- x@coords # true coordinate of the measurement (rather than the BAU centroids)
split_df <- group_split(x@data, id)
names(split_df) <- sapply(split_df, function(df) df$id[1])

# Only consider cells with at least 30 observations
# sum(table(df$cell) < 30)
# sum(table(df$cell) >= 30)
idx <- which(sapply(split_df, function(x) nrow(x) >= 10))
split_df <- split_df[idx]


# ---- Plot the raw data ----

draw_world_custom <- function(g) {

  ## Load the world map data from the FRK package
  data(worldmap, envir=environment(), package = "FRK")

  ## Homogenise (see details) to avoid lines crossing the map
  worldmap <- FRK:::.homogenise_maps(worldmap)

  ## Now return a gg object with the map overlayed
  g + geom_polygon(data = worldmap, aes(x=long, y=lat, group=group), fill="black", size=0.1)
}

## A palette for the data and predictions
nasa_palette <- c(
  "#03006d","#02008f","#0000b6","#0001ef","#0000f6","#0428f6","#0b53f7",
           "#0f81f3","#18b1f5","#1ff0f7","#27fada","#3efaa3","#5dfc7b","#85fd4e",
           "#aefc2a","#e9fc0d","#f6da0c","#f5a009","#f6780a","#f34a09","#f2210a",
           "#f50008","#d90009","#a80109","#730005"
)

# Brazil-Malvinas confluence zone and Ocean region
BM_box <- cbind(lon = c(-60, -48), lat = c(-49, -35))
Ocean_box <- cbind(lon = c(76, 88), lat = c(-31, -17))

df$Z_clipped <- pmin(pmax(df$Z, -8), 8)
Zplot <- plot_spatial_or_ST(df, column_names = "Z_clipped", plot_over_world = T, pch = 46)[[1]]
Zplot <- draw_world_custom(Zplot)
Zplot <- Zplot +
  scale_colour_gradientn(colours = nasa_palette) +
  labs(colour = expression(bold(Z)~(degree*C))) +
  theme(axis.title = element_blank()) +
  theme(panel.border = element_blank(),
        panel.background = element_blank())

Zplot <- Zplot +
  annotate("rect", xmin = BM_box[1, "lon"], xmax = BM_box[2, "lon"], ymin = BM_box[1, "lat"], ymax = BM_box[2, "lat"], fill=NA, color="red", size=1) +
  annotate("rect", xmin = Ocean_box[1, "lon"], xmax = Ocean_box[2, "lon"], ymin = Ocean_box[1, "lat"], ymax = Ocean_box[2, "lat"], fill=NA, color="red", size=1)

ggsave(
  Zplot,
  filename = "data.png", device = "png", width = 6, height = 2.8,
  path = img_path
)

map_layer <- geom_map(
  data = map_data("world"),
  map = map_data("world"),
  aes(group = group, map_id = region),
  fill = "black", colour = "black", size = 0.1
)

BMconfluence <- ggplot() +
  scale_colour_gradientn(colours = nasa_palette, name = expression(degree*C)) +
  xlab("Longitude (deg)") + ylab("Latitude (deg)") +
  map_layer +
  xlim(BM_box[, "lon"]) +
  ylim(BM_box[, "lat"]) +
  theme_bw() +
  coord_fixed(expand = FALSE) +
  geom_point(data = df_backup, aes(lon, lat, colour =  pmin(pmax(Z, -8), 8)), pch = 46)

Ocean <- ggplot() +
  scale_colour_gradientn(colours = nasa_palette, name = expression(degree*C)) +
  xlab("Longitude (deg)") + ylab("Latitude (deg)") +
  map_layer +
  xlim(Ocean_box[, "lon"]) +
  ylim(Ocean_box[, "lat"]) +
  theme_bw() +
  coord_fixed(expand = FALSE) +
  geom_point(data = df_backup, aes(lon, lat, colour =  pmin(pmax(Z, -8), 8)), pch = 46)

ggsave(
  ggarrange(Zplot, BMconfluence, Ocean, common.legend = T, nrow = 1, ncol = 3, legend = "right"),
  filename = "data_highlights.png", device = "png", width = 9, height = 2.5,
  path = img_path
)

# ---- Estimation ----

estimator = juliaLet('
  include(joinpath(pwd(), "src/architecture.jl"))
  estimator = gnnarchitecture(p)
                      ', p = p)

estimator <- loadbestweights(estimator, "intermediates/application/SST/runs_pointestimator")

ciestimator = juliaLet('
  intervalestimator = IntervalEstimator(deepcopy(estimator), deepcopy(estimator))
                      ', estimator = estimator)

ciestimator <- loadbestweights(ciestimator, "intermediates/application/SST/runs_CIestimator")

estimate_parameters <- function(estimator, ciestimator, dat) {

  # Convert data into correct form (n x m matrix, where n is the number of
  # observations and m is the number of replicates, here equal to 1).
  # Also centre the data around zero.
  Z <- matrix(dat$Z, nrow = 1)
  Z <- Z - mean(Z)


  # Spatial distance matrix
  S <- as.matrix(dat[, c("lon", "lat")])
  colnames(S) <- NULL
  S <- chord_length(S)

  # Scale the distances so that they are between 0 and sqrt(2)
  scale_factor <- sqrt(2) / (max(S) - min(S))
  S <- (S-min(S)) * scale_factor

  # Construct the graph
  g <- juliaLet("A = adjacencymatrix(S, 0.15); GNNGraph(A,  ndata = Z)", S = S, Z = Z)

  thetahat   <- estimate(estimator, g)
  thetahatci <- estimate(ciestimator, g)

  # inverse of scale transformation to range parameter
  thetahat[2, ] <- thetahat[2, ] / scale_factor
  thetahatci[2, ] <- thetahatci[2, ] / scale_factor
  thetahatci[5, ] <- thetahatci[5, ] / scale_factor

  # Put estimates into a convenient data frame
  estimates <- rbind(thetahat, thetahatci)
  rownames(estimates) <- c("tau", "rho", "sigma", "tau_lower", "rho_lower", "sigma_lower", "tau_upper", "rho_upper", "sigma_upper")

  return(estimates)
}



# ---- Data prep ----

# Convert data into correct form (n x m matrix, where n is the number of
# observations and m is the number of replicates, here equal to 1)
Z <- lapply(split_df, function(x) matrix(x$Z, nrow = 1))
Z <- lapply(Z, function(z) z - mean(z)) # centre the data around zero

# Spatial distance matrix
S <- lapply(split_df, function(x) {
  s <- as.matrix(x[, c("lon", "lat")])
  colnames(s) <- NULL
  chord_length(s)
})

# Scale the distances so that they are between 0 and sqrt(2)
scales <- sapply(S, function(s) {
  sqrt(2) / (max(s) - min(s))
})

S <- lapply(seq_along(S), function(i) {
  s <- S[[i]]
  (s-min(s)) * scales[i]
})


# Construct the graphs
g <- lapply(S, function(s) {
  juliaLet("A = adjacencymatrix(S, 0.15); GNNGraph(A)", S = s)
})

# Add observed data to the graphs
Zgraph <- lapply(seq_along(g), function(i) {
  juliaLet("
           GNNGraph(g; ndata = Z)
           ", g = g[[i]], Z = Z[[i]])
})


# ---- Apply the estimator ----

thetahat   <- estimate(estimator, Zgraph)
thetahatci <- estimate(ciestimator, Zgraph)

# inverse of scale transformation to range parameter
# thetahat[2, ] <- thetahat[2, ] / scales
# thetahatci[2, ] <- thetahatci[2, ] / scales
# thetahatci[4, ] <- thetahatci[4, ] / scales
thetahat[2, ] <- thetahat[2, ] / scales
thetahatci[2, ] <- thetahatci[2, ] / scales
thetahatci[5, ] <- thetahatci[5, ] / scales


# ---- Plot the estimates

# Put estimates into a data frame for merging into BAU object
estimates <- rbind(thetahat, thetahatci)
colnames(estimates) <- names(split_df)
rownames(estimates) <- c("tau", "rho", "sigma", "tau_lower", "rho_lower", "sigma_lower", "tau_upper", "rho_upper", "sigma_upper")
estimates <- melt(estimates, varnames = c("parameter", "id"), value.name = "estimate")

# merge estimates into bau object
rho <- merge(baus, filter(estimates, parameter == "rho"))
rho_plot <- plot_spatial_or_ST(rho, column_names = "estimate", plot_over_world = T)[[1]]
rho_plot <- draw_world_custom(rho_plot)
rho_plot <-
  rho_plot +
  scale_fill_distiller(palette = "YlOrRd", na.value = NA, direction = 1) +
  labs(fill = expression(hat(rho))) +
  theme(axis.title = element_blank())

sigma <- merge(baus, filter(estimates, parameter == "sigma"))
sigma_plot <- plot_spatial_or_ST(sigma, column_names = "estimate", plot_over_world = T)[[1]]
sigma_plot <- draw_world_custom(sigma_plot)
sigma_plot <-
  sigma_plot +
  scale_fill_distiller(palette = "YlOrRd", na.value = NA, direction = 1) +
  labs(fill = expression(hat(sigma))) +
  theme(axis.title = element_blank())

# merge estimates into bau object
tau <- merge(baus, filter(estimates, parameter == "tau"))
tau@data$pminestimate <- pmin(0.5, tau@data$estimate)
tau_plot <- plot_spatial_or_ST(tau, column_names = "pminestimate", plot_over_world = T)[[1]]
tau_plot <- draw_world_custom(tau_plot)
tau_plot <- tau_plot +
  scale_fill_distiller(palette = "YlOrRd", na.value = NA, direction = 1) +
  labs(fill = expression(hat(tau))) +
  theme(axis.title = element_blank())

ggsave(
  ggpubr::ggarrange(rho_plot, sigma_plot, tau_plot, align = "hv", nrow = 1, legend = "top"),
  filename = "estimates.pdf", device = "pdf", width = 13, height = 4,
  path = img_path
)


# ---- Plot the credible intervals ----

plot_estimates <- function(baus, estimates, param, limits) {

  baus <- merge(baus, filter(estimates, parameter == param))

  gg <- plot_spatial_or_ST(baus, column_names = "estimate", plot_over_world = T)[[1]]
  gg <- draw_world_custom(gg)
  gg <- gg +
    scale_fill_distiller(palette = "YlOrRd", na.value = NA, limits = limits, direction = 1) +
    labs(fill = "") +
    theme(axis.title = element_blank())
  gg
}

# Put estimates into a data frame for merging into BAU object
estimates <- rbind(thetahat, thetahatci)
colnames(estimates) <- names(split_df)
rownames(estimates) <- c("tau", "rho", "sigma", "tau_lower", "rho_lower", "sigma_lower", "tau_upper", "rho_upper", "sigma_upper")
estimates <- melt(estimates, varnames = c("parameter", "id"), value.name = "estimate")

limits <- estimates %>% filter(parameter %in% c("rho_lower", "rho_upper")) %>% summarise(range(estimate))
limits <- limits[[1]]
rho_lower <- plot_estimates(baus, estimates, "rho_lower", limits) + labs(title = expression(hat(rho) *": lower bound" ))
rho_upper <- plot_estimates(baus, estimates, "rho_upper", limits) + labs(title = expression(hat(rho) *": upper bound"))
rho_ci <- ggpubr::ggarrange(rho_lower, rho_upper, align = "hv", nrow = 1, legend = "right", common.legend = T)

limits <- estimates %>% filter(parameter %in% c("sigma_lower", "sigma_upper")) %>% summarise(range(estimate))
limits <- limits[[1]]
sigma_lower <- plot_estimates(baus, estimates, "sigma_lower", limits) + labs(title = expression(hat(sigma) *": lower bound" ))
sigma_upper <- plot_estimates(baus, estimates, "sigma_upper", limits) + labs(title = expression(hat(sigma) *": upper bound"))
sigma_ci <- ggpubr::ggarrange(rho_lower, rho_upper, align = "hv", nrow = 1, legend = "right", common.legend = T)

estimates <- estimates %>%
  filter(parameter %in% c("tau_lower", "tau_upper")) %>%
  mutate(estimate = pmin(estimate, 0.5))
limits <- estimates %>% summarise(range(estimate))
limits <- limits[[1]]
tau_lower <- plot_estimates(baus, estimates, "tau_lower", limits) + labs(title = expression(hat(tau) *": lower bound"))
tau_upper <- plot_estimates(baus, estimates, "tau_upper", limits) + labs(title = expression(hat(tau) *": upper bound"))
tau_ci <- ggpubr::ggarrange(tau_lower, tau_upper, align = "hv", nrow = 1, legend = "right", common.legend = T)

ggsave(
  ggpubr::ggarrange(rho_ci, sigma_ci, tau_ci, ncol = 1),
  filename = "intervals.pdf", device = "pdf", width = 8, height = 7,
  path = img_path
)


# ---- Neighbours ----

nb <- poly2nb(baus)

# number of neighbours for each bau
table(card(nb))

subset_baus <- function(baus, i, neighbour_list) {
  idx <- c(i, neighbour_list[[i]])
  bau_subset <- baus@polygons[idx]
  bau_subset <- SpatialPolygons(bau_subset, 1:length(bau_subset), baus@proj4string)
  newdata    <- data.frame(central_hexagon = as.numeric(idx == i))
  row.names(newdata) <- sapply(bau_subset@polygons, slot, "ID")
  bau_subset <- SpatialPolygonsDataFrame(bau_subset, data = newdata)
  return(bau_subset)
}

# Sanity check: plot some BAUs and their neighbours
bau_subset1 <- subset_baus(baus, 400, nb)
bau_subset2 <- subset_baus(baus, 800, nb)
bau_subset <- rbind(bau_subset1, bau_subset2)
gg <- plot_spatial_or_ST(bau_subset, "central_hexagon", plot_over_world = T)[[1]]
gg <- gg %>% draw_world_custom
# gg + labs(fill = "Central hexagon", x = "", y = "")
gg + theme(legend.position = "none", axis.title = element_blank())

# Now create a list of hexagon clusters, each associated with a central hexagon.
# We just need to store the coordinates and data as a data frame.
N == length(nb) # sanity check
clustered_split_df <- lapply(1:N, function(i) {
  idx <- c(i, nb[[i]]) # get the neighbours of the ith BAU
  idx <- as.character(idx)
  # extract the data frames for each neighbour
  dat <- lapply(idx[idx %in% names(split_df)], function(y) split_df[[y]])
  dat <- do.call(rbind, dat)
  dat
})
names(clustered_split_df) <- 1:N # BAU id

# Sanity check: plot the data and the corresponding hexagon cluster to ensure that they overlap
i <- 400
tmp <- clustered_split_df[[i]]
tmp <- SpatialPointsDataFrame(tmp[, c("lon", "lat")], data = tmp[, "Z"])
plot_spatial_or_ST(tmp, "Z", plot_over_world = T)[[1]]
plot_spatial_or_ST(subset_baus(baus, i, nb), "central_hexagon", plot_over_world = T)[[1]]

clustered_split_df <- clustered_split_df[!sapply(clustered_split_df, is.null)]
estimates <- sapply(clustered_split_df, estimate_parameters)
estimates_backup <- estimates
rownames(estimates) <- c("tau", "rho", "sigma", "tau_lower", "rho_lower", "sigma_lower", "tau_upper", "rho_upper", "sigma_upper")

tau_ciwidth <- as.numeric(estimates["tau_upper", ] - estimates["tau_lower", ])
rho_ciwidth <- as.numeric(estimates["rho_upper", ] - estimates["rho_lower", ])
sigma_ciwidth <- as.numeric(estimates["sigma_upper", ] - estimates["sigma_lower", ])
ciwidth <- matrix(c(tau_ciwidth, rho_ciwidth, sigma_ciwidth), nrow = 3, byrow = T)
rownames(ciwidth) <- c("tau_ciwidth", "rho_ciwidth", "sigma_ciwidth")
estimates <- rbind(estimates, ciwidth)

colnames(estimates) <- names(clustered_split_df)
estimates <- melt(estimates, varnames = c("parameter", "id"), value.name = "estimate")

# Plot each parameter estimate
plot_estimates <- function(baus, estimates, param) {

  baus <- merge(baus, filter(estimates, parameter == param))

  gg <- plot_spatial_or_ST(baus, column_names = "estimate", plot_over_world = T)[[1]]
  gg <- draw_world_custom(gg)
  gg <- gg +
    scale_fill_distiller(palette = "YlOrRd", na.value = NA, direction = 1) +
    theme(axis.title = element_blank(),
          panel.border = element_blank(),
          panel.background = element_blank(),
          legend.position = "top",
          legend.key.width = unit(1, 'cm'))
  gg
}

rho_plot   <- plot_estimates(baus, estimates, "rho") + labs(fill = expression(hat(rho)))
sigma_plot <- plot_estimates(baus, estimates, "sigma") + labs(fill = expression(hat(sigma)))
tau_plot   <- plot_estimates(baus, mutate(estimates, estimate = pmin(estimate, 0.6)), "tau") + labs(fill = expression(hat(tau)))

plot_ciwidth <- function(baus, estimates, param) {

  baus <- merge(baus, filter(estimates, parameter == param))

  gg <- plot_spatial_or_ST(baus, column_names = "estimate", plot_over_world = T)[[1]]
  gg <- draw_world_custom(gg)
  gg <- gg +
    # scale_fill_distiller(palette = "BrBG", na.value = NA, direction = -1) + # white centre value conflicts with missing colour (also white)
    scale_fill_distiller(palette = "GnBu", na.value = NA, direction = -1) +
    theme(axis.title = element_blank(),
          panel.border = element_blank(),
          panel.background = element_blank(),
          legend.position = "top",
          legend.key.width = unit(1, 'cm'))
  gg
}

rhoci_plot   <- plot_ciwidth(baus, estimates, "rho_ciwidth") + labs(fill = expression(atop(hat(rho) ~ "credible-", "interval width")))
sigmaci_plot <- plot_ciwidth(baus, estimates, "sigma_ciwidth") + labs(fill = expression(atop(hat(sigma) ~ "credible-", "interval width")))
tauci_plot   <- plot_ciwidth(baus, mutate(estimates, estimate = pmin(estimate, 0.5)), "tau_ciwidth") + labs(fill = expression(atop(hat(tau) ~ "credible-", "interval width")))

ggsave(
  ggpubr::ggarrange(
    rho_plot, sigma_plot, tau_plot,
    rhoci_plot, sigmaci_plot, tauci_plot,
    align = "hv", nrow = 2, ncol = p),
  filename = "estimates_clustered.pdf", device = "pdf", width = 14, height = 6,
  path = img_path
)
