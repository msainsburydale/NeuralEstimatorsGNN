library("optparse")
option_list = list(
  make_option(c("-q", "--quick"), type = "logical", default = FALSE, action = "store_true")
)
opt_parser  <- OptionParser(option_list=option_list)
quick       <- parse_args(opt_parser)$quick

suppressMessages({
  library("NeuralEstimators")
  library("JuliaConnectoR")
  library("reshape2")
  library("ggplot2")
  library("dggrids")
  library("dplyr")
  library("FRK")
  library("sp")
  library("ggpubr")
  library("spdep") # poly2nb()
  library("spatstat.geom") # nndist()
  options(dplyr.summarise.inform = FALSE)
})

img_path <- "img/application/SST"
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

p = 3L # number of parameters


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

## Remove impossible locations, and remove repetitions
df <- df %>%
  subset(lon < 180 & lon > -180 & lat < 90 & lat > -90) %>%
  distinct(lon, lat, .keep_all = TRUE)

## Detrend the data following Zammit-Mangion and Rougier (2020).
## Here, we define the residuals from a linear model, which will be used as
## the response variable for the analysis.
df$lat2 <- df$lat^2
df$Z    <- residuals(lm(sst ~ 1 + lat + lat2, data = df))
df$sst  <- df$lat2 <- NULL

# ---- Plot the raw data ----

draw_world_custom <- function(g) {

  ## Load the world map data from the FRK package
  data(worldmap, envir=environment(), package = "FRK")

  ## Homogenise (see details) to avoid lines crossing the map
  worldmap <- FRK:::.homogenise_maps(worldmap)

  ## Now return a gg object with the map overlayed
  g + geom_polygon(data = worldmap, aes(x=long, y=lat, group=group), fill="black", linewidth=0.1)
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

spdf <- df
coordinates(spdf) = ~ lon + lat
slot(spdf, 'proj4string') <- CRS('+proj=longlat +ellps=sphere')

spdf$Z_clipped <- pmin(pmax(spdf$Z, -8), 8)
Zplot <- plot_spatial_or_ST(spdf, column_names = "Z_clipped", plot_over_world = T, pch = 46)[[1]]
Zplot <- draw_world_custom(Zplot)
suppressMessages({
  Zplot <- Zplot +
    scale_colour_gradientn(colours = nasa_palette) +
    labs(colour = expression(bold(Z)~(degree*C))) +
    theme(axis.title = element_blank()) +
    theme(panel.border = element_blank(),
          panel.background = element_blank())
})


Zplot <- Zplot +
  annotate("rect", xmin = BM_box[1, "lon"], xmax = BM_box[2, "lon"], ymin = BM_box[1, "lat"], ymax = BM_box[2, "lat"], fill=NA, color="red", linewidth=1) +
  annotate("rect", xmin = Ocean_box[1, "lon"], xmax = Ocean_box[2, "lon"], ymin = Ocean_box[1, "lat"], ymax = Ocean_box[2, "lat"], fill=NA, color="red", linewidth=1)

ggsave(
  Zplot,
  filename = "data.png", device = "png", width = 6, height = 2.8,
  path = img_path
)

map_layer <- geom_map(
  data = map_data("world"),
  map = map_data("world"),
  aes(group = group, map_id = region),
  fill = "black", colour = "black", linewidth = 0.1
)

BMconfluence <- ggplot() +
  scale_colour_gradientn(colours = nasa_palette, name = expression(degree*C)) +
  xlab("Longitude (deg)") + ylab("Latitude (deg)") +
  map_layer +
  xlim(BM_box[, "lon"]) +
  ylim(BM_box[, "lat"]) +
  theme_bw() +
  coord_fixed(expand = FALSE) +
  geom_point(data = df, aes(lon, lat, colour =  pmin(pmax(Z, -8), 8)), pch = 46)

Ocean <- ggplot() +
  scale_colour_gradientn(colours = nasa_palette, name = expression(degree*C)) +
  xlab("Longitude (deg)") + ylab("Latitude (deg)") +
  map_layer +
  xlim(Ocean_box[, "lon"]) +
  ylim(Ocean_box[, "lat"]) +
  theme_bw() +
  coord_fixed(expand = FALSE) +
  geom_point(data = df, aes(lon, lat, colour =  pmin(pmax(Z, -8), 8)), pch = 46)

suppressWarnings({
  ggsave(
    ggarrange(Zplot, BMconfluence, Ocean, common.legend = T, nrow = 1, ncol = 3, legend = "right"),
    filename = "data_highlights.png", device = "png", width = 9, height = 2.5,
    path = img_path
  )
})


# ---- Hexagon cell pre-processing ----

## Bin the data into hexagons
spdf <- if (quick) df[sample(1:nrow(df), 50000), ] else df
coordinates(spdf) = ~ lon + lat
suppressWarnings({
  slot(spdf, 'proj4string') <- CRS('+proj=longlat +ellps=sphere')
})
baus <- auto_BAUs(manifold = sphere(), type = "hex", isea3h_res = 5, data = spdf)

# relabel the BAUs to be from 1:N (by default, there are some missing numbers
# which can cause problems)
N <- length(baus)
for (i in 1:length(baus)) {
  baus@polygons[[i]]@ID <- as.character(i)
}
names(baus@polygons) <- 1:N
baus@data$id <- 1:N

x <- map_to_BAUs(spdf, sp_pols = baus)
x@data[, c("lon", "lat")] <- x@coords # true coordinate of the measurement (rather than the BAU centroids)
split_df <- group_split(x@data, id)
names(split_df) <- sapply(split_df, function(df) df$id[1])

# Only consider cells with at least 30 observations
idx <- which(sapply(split_df, function(x) nrow(x) >= 30))
split_df <- split_df[idx]

## Hexagon clusters
suppressMessages(suppressWarnings({
  sf_use_s2(FALSE) # https://github.com/r-spatial/sf/issues/1762#issuecomment-900571711
  nb <- poly2nb(baus)
}))


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
gg <- draw_world_custom(gg)
gg <- gg + theme(legend.position = "none", axis.title = element_blank(), panel.border = element_blank(), panel.background = element_blank())
ggsave(gg, filename = "clustering.pdf", device = "pdf", width = 6.8, height = 3.8, path = img_path)
ggsave(gg, filename = "clustering.png", device = "png", width = 6.8, height = 3.8, path = img_path)

# Now create a list of hexagon clusters, each associated with a central hexagon.
# We just need to store the coordinates and data as a data frame.
# N == length(nb) # sanity check
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
gg1 <- plot_spatial_or_ST(tmp, "Z", plot_over_world = T)[[1]]
gg2 <- plot_spatial_or_ST(subset_baus(baus, i, nb), "central_hexagon", plot_over_world = T)[[1]]
# ggarrange(gg1, gg2)


## Remove null entries
clustered_split_df <- clustered_split_df[!sapply(clustered_split_df, is.null)]

# Summarise the number of observed locations, n, across the clusters:
# length(clustered_split_df) # 2161: number of clusters that we are making inference over
n <- sapply(clustered_split_df, nrow)
# hist(n)
# summary(n)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# 30    1081    2769    3204    4976   12591

# histogram of sample sizes
gghist <- ggplot() +
  geom_histogram(aes(x = n), bins = 20, fill = "gray", colour= "black") +
  labs(x = expression("Number of observed locations,"~italic(n)), y = "Frequency") +
  theme_bw()
ggsave(gghist, filename = "samplesizes.pdf", device = "pdf", width = 6.8, height = 3.8, path = img_path)
ggsave(gghist, filename = "samplesizes.png", device = "png", width = 6.8, height = 3.8, path = img_path)

# Combined figure of cells and sample sizes
ggsave(ggarrange(gg, gghist), filename = "clusteringsamplesizes.pdf", device = "pdf", width = 9, height = 3, path = img_path)
ggsave(ggarrange(gg, gghist), filename = "clusteringsamplesizes.png", device = "png", width = 9, height = 3, path = img_path)


# ---- Estimation ----

# Start Julia with the project of the current directory:
Sys.setenv("JULIACONNECTOR_JULIAOPTS" = "--project=. --threads=auto")

# Load the architecture and prior information (which is incorporated in the architecture)
juliaEval('
  include(joinpath(pwd(), "src/architecture.jl"))
  Ω = (
	  τ = Uniform(0.1, 1.0),
	  ρ = Uniform(0.05, 0.6),
	  σ = Uniform(0.1, 3.0)
  )
  a = [minimum.(values(Ω))...]
  b = [maximum.(values(Ω))...]
  ')

# Initialise the estimators
estimator   = juliaLet('estimator = gnnarchitecture(p)', p = p)
ciestimator = juliaLet('
  U = gnnarchitecture(p; final_activation = identity)
  V = deepcopy(U)
  intervalestimator = IntervalEstimatorCompactPrior(U, V, a, b))
  ', p = p)

# Load the optimal weights
estimator   <- loadbestweights(estimator, "intermediates/application/SST/runs_pointestimator")
ciestimator <- loadbestweights(ciestimator, "intermediates/application/SST/runs_CIestimator")

estimate_parameters <- function(estimator, ciestimator, dat) {

  # Convert data into correct form (n x m matrix, where n is the number of
  # observations and m is the number of replicates, here equal to 1).
  # Also centre the data around zero.
  Z <- matrix(dat$Z, nrow = 1)
  Z <- Z - mean(Z)

  # Matrix of spatial locations
  S <- as.matrix(dat[, c("lon", "lat")])
  colnames(S) <- NULL
  n <- nrow(S)


  preprocessing_time <<- preprocessing_time + system.time({

    # Convert from (lon, lat) to (x, y, z)
    S <- t(apply(S, 1, xyz_conversion))

    # Compute the scale factor use to scale distances between [0, sqrt(2)]
    max_dist <- max(dist(S[chull(S), ]))
    min_dist <- min(nndist(S))
    scale_factor <- sqrt(2) / (max_dist - min_dist)

    # Compute the (sparse) adjacency matrix quickly
    # https://stackoverflow.com/a/47690594
    # See also: https://cran.r-project.org/web/packages/N2R/N2R.pdf
    r0 <- 0.15                # fixed radius used during training on the unit square
    r  <- r0 / scale_factor   # neighbourhood disc radius used here
    k  <- 30L                 # maximum number of neighbours to consider

    # Construct the graph
    g = juliaLet('
    # Compute the adjacency matrix
    A = adjacencymatrix(S, r, k)

    # scale the distances so that they are between [0, sqrt(2)]
    v = A.nzval
    v .-=  min_dist
    v .*= scale_factor

    # construct the graph
    g = GNNGraph(A, ndata = Z)
    g
    ', S=S, r=r, k=k, Z=Z, min_dist=min_dist, scale_factor=scale_factor)

  })["elapsed"]

  # ---- Estimate

  # Super-assignment to keep track of the estimation time
  estimation_time <<- estimation_time + system.time({
    thetahat   <- estimate(estimator, g)
    thetahatci <- estimate(ciestimator, g)
  })["elapsed"]

  # inverse of scale transformation to range parameter
  thetahat[2, ] <- thetahat[2, ] / scale_factor
  thetahatci[2, ] <- thetahatci[2, ] / scale_factor
  thetahatci[5, ] <- thetahatci[5, ] / scale_factor

  # Put estimates into a convenient data frame
  estimates <- rbind(thetahat, thetahatci)
  rownames(estimates) <- c("tau", "rho", "sigma", "tau_lower", "rho_lower", "sigma_lower", "tau_upper", "rho_upper", "sigma_upper")

  return(estimates)
}

# Dummy run to compile the Julia code (more accurate timings)
estimation_time <- preprocessing_time <- 0
estimates <- sapply(clustered_split_df[1:3], function(dat) {
  estimate_parameters(estimator, ciestimator, dat)
})


# ---- Estimation ----

# NB just replace "clustered_split_df" with "split_df" to generate estimates without the clustering approach

estimation_time <- preprocessing_time <- 0

cat("\nConstructing the adjacency matrices and estimating... \n")

## non-parallel version
estimates <- sapply(clustered_split_df, function(dat) {
  estimate_parameters(estimator, ciestimator, dat)
})
## parallel version
# estimates <- estimate_parameters_parallel(estimator, ciestimator, clustered_split_df)

write.csv(preprocessing_time, paste0(img_path, "/preprocessing_time.csv"))
write.csv(estimation_time, paste0(img_path, "/estimation_time.csv"))

# estimates_backup <- estimates
rownames(estimates) <- c("tau", "rho", "sigma", "tau_lower", "rho_lower", "sigma_lower", "tau_upper", "rho_upper", "sigma_upper")

tau_ciwidth <- as.numeric(estimates["tau_upper", ] - estimates["tau_lower", ])
rho_ciwidth <- as.numeric(estimates["rho_upper", ] - estimates["rho_lower", ])
sigma_ciwidth <- as.numeric(estimates["sigma_upper", ] - estimates["sigma_lower", ])
ciwidth <- matrix(c(tau_ciwidth, rho_ciwidth, sigma_ciwidth), nrow = 3, byrow = T)
rownames(ciwidth) <- c("tau_ciwidth", "rho_ciwidth", "sigma_ciwidth")
estimates <- rbind(estimates, ciwidth)

colnames(estimates) <- names(clustered_split_df)
estimates <- melt(estimates, varnames = c("parameter", "id"), value.name = "estimate")

# Plot the point estimates
plot_estimates <- function(baus, estimates, param, limits = c(NA, NA)) {

  baus <- merge(baus, filter(estimates, parameter == param))

  suppressMessages({

    gg <- plot_spatial_or_ST(baus, column_names = "estimate", plot_over_world = T)[[1]]
    gg <- draw_world_custom(gg)
    gg <- gg +
      scale_fill_distiller(palette = "YlOrRd", na.value = NA, direction = 1, limits = limits) +
      theme(axis.title = element_blank(),
            panel.border = element_blank(),
            panel.background = element_blank(),
            legend.position = "top",
            legend.key.width = unit(1, 'cm'))

  })

  gg
}

rho_plot   <- plot_estimates(baus, estimates, "rho") + labs(fill = expression(hat(rho)))
sigma_plot <- plot_estimates(baus, estimates, "sigma") + labs(fill = expression(hat(sigma)))
tau_plot   <- plot_estimates(baus, mutate(estimates, estimate = pmin(estimate, 0.6)), "tau") + labs(fill = expression(hat(tau)))

# Plot the credible interval widths
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
  filename = "estimates.pdf", device = "pdf", width = 14, height = 6,
  path = img_path
)


# Plot the lower and upper quantiles
limits <- estimates %>% filter(parameter %in% c("rho_lower", "rho_upper")) %>% summarise(range(estimate))
limits <- limits[[1]]
rho_lower <- plot_estimates(baus, estimates, "rho_lower", limits) + labs(title = expression(hat(rho) *": lower bound" ), fill = "")
rho_upper <- plot_estimates(baus, estimates, "rho_upper", limits) + labs(title = expression(hat(rho) *": upper bound"), fill = "")
rho_ci <- ggpubr::ggarrange(rho_lower, rho_upper, align = "hv", nrow = 1, legend = "right", common.legend = T)

limits <- estimates %>% filter(parameter %in% c("sigma_lower", "sigma_upper")) %>% summarise(range(estimate))
limits <- limits[[1]]
sigma_lower <- plot_estimates(baus, estimates, "sigma_lower", limits) + labs(title = expression(hat(sigma) *": lower bound" ), fill = "")
sigma_upper <- plot_estimates(baus, estimates, "sigma_upper", limits) + labs(title = expression(hat(sigma) *": upper bound"), fill = "")
sigma_ci <- ggpubr::ggarrange(sigma_lower, sigma_upper, align = "hv", nrow = 1, legend = "right", common.legend = T)

estimates_tau <- estimates %>%
  filter(parameter %in% c("tau_lower", "tau_upper")) %>%
  mutate(estimate = pmin(estimate, 0.6))
limits <- estimates_tau %>% summarise(range(estimate))
limits <- limits[[1]]
tau_lower <- plot_estimates(baus, estimates_tau, "tau_lower", limits) + labs(title = expression(hat(tau) *": lower bound"), fill = "")
tau_upper <- plot_estimates(baus, estimates_tau, "tau_upper", limits) + labs(title = expression(hat(tau) *": upper bound"), fill = "")
tau_ci <- ggpubr::ggarrange(tau_lower, tau_upper, align = "hv", nrow = 1, legend = "right", common.legend = T)

ggsave(
  ggpubr::ggarrange(rho_ci, sigma_ci, tau_ci, ncol = 1),
  filename = "intervals.png", device = "png", width = 8, height = 7,
  path = img_path
)

ggsave(
  ggpubr::ggarrange(rho_ci, sigma_ci, tau_ci, ncol = 1),
  filename = "intervals.pdf", device = "pdf", width = 8, height = 7,
  path = img_path
)

# --------

# TODO parallel estimation to improve estimation times even further

# NB just sticking with non-parallel version for now because it's a bit simpler
estimate_parameters_parallel <- function(estimator, ciestimator, split_df_list) {

  preprocessing_time <<- preprocessing_time + system.time({

  # Convert data into correct form (n x m matrix, where n is the number of
  # observations and m is the number of replicates, here equal to 1).
  # Also centre the data around zero.
  Z <- lapply(split_df_list, function(x) matrix(x$Z, nrow = 1))
  Z <- lapply(Z, function(z) z - mean(z)) # centre the data around zero

  # Matrix of spatial locations, converted from (lon, lat) to (x, y, z)
  S <- lapply(split_df_list, function(x) {
    S <- as.matrix(x[, c("lon", "lat")])
    colnames(S) <- NULL
    S <- t(apply(S, 1, xyz_conversion))
    S
  })

  # Compute the scale factor use to scale distances between [0, sqrt(2)]
  max_dist <- as.numeric(sapply(S, function(S) max(dist(S[chull(S), ]))))
  min_dist <- as.numeric(sapply(S, function(S) min(nndist(S))))
  scale_factor <- sqrt(2) / (max_dist - min_dist)

  # Hyperparameters for the adjacency matrix (i.e., the neighbourhood definition)
  r0 <- 0.15                # fixed radius used during training on the unit square
  r  <- r0 / scale_factor   # neighbourhood disc radius used here
  k  <- 30L                 # maximum number of neighbours to consider

  # Construct the graphs
  g = lapply(seq_along(S), function(i) {
    juliaLet('
      # Compute the adjacency matrix
      A = adjacencymatrix(S, r, k)

      # scale the distances so that they are between [0, sqrt(2)]
      v = A.nzval
      v .-=  min_dist
      v .*= scale_factor

      # construct the graph
      g = GNNGraph(A, ndata = Z)
      g
      ', S=S[[i]], r=r[i], k=k, Z=Z[[i]], min_dist=min_dist[i], scale_factor=scale_factor[i])
  })

  })["elapsed"]

  # Apply the estimators
  estimation_time <<- estimation_time + system.time({
  thetahat   <- estimate(estimator, g)
  thetahatci <- estimate(ciestimator, g)
  })["elapsed"]

  # inverse of scale transformation to range parameter
  thetahat[2, ] <- thetahat[2, ] / scale_factor
  thetahatci[2, ] <- thetahatci[2, ] / scale_factor
  thetahatci[5, ] <- thetahatci[5, ] / scale_factor

  # Put estimates into a convenient data frame
  estimates <- rbind(thetahat, thetahatci)
  rownames(estimates) <- c("tau", "rho", "sigma", "tau_lower", "rho_lower", "sigma_lower", "tau_upper", "rho_upper", "sigma_upper")

  return(estimates)
}
