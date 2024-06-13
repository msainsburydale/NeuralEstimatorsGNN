library("optparse")
option_list = list(
  make_option(c("-q", "--quick"), type = "logical", default = FALSE, action = "store_true")
)
opt_parser  <- OptionParser(option_list=option_list)
quick       <- parse_args(opt_parser)$quick

options(warn = -1)

suppressMessages({
  library("NeuralEstimators")
  library("JuliaConnectoR")
  library("reshape2")
  library("ggplot2")
  library("GpGp")
  library("Matrix")
  library("dggrids")
  library("dplyr")
  library("FRK")
  library("sp")
  library("sf")
  library("ggpubr")
  library("parallel") # mclapply
  library("spdep") # poly2nb()
  library("spatstat.geom") # nndist()
  library("nabor")
  options(dplyr.summarise.inform = FALSE)
})

source("src/plotting.R")

img_path <- "img/application"
int_path <- "intermediates/application"
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)
dir.create(int_path, recursive = TRUE, showWarnings = FALSE)

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

# ---- Plot the data ----

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

Zplot_norect <- Zplot
Zplot <- Zplot +
  annotate("rect", xmin = BM_box[1, "lon"], xmax = BM_box[2, "lon"], ymin = BM_box[1, "lat"], ymax = BM_box[2, "lat"], fill=NA, color="red", linewidth=1) +
  annotate("rect", xmin = Ocean_box[1, "lon"], xmax = Ocean_box[2, "lon"], ymin = Ocean_box[1, "lat"], ymax = Ocean_box[2, "lat"], fill=NA, color="red", linewidth=1)

ggsv(Zplot, filename = "data", width = 6, height = 2.8, path = img_path)

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

Zplot_insets <-ggpubr::ggarrange(Zplot, BMconfluence, Ocean, common.legend = T, nrow = 1, ncol = 3)

ggsv(Zplot_insets, filename = "data_highlights", width = 9, height = 2.5, path = img_path)


# ---- Hexagon cell pre-processing ----

## Bin the data into cells
spdf <- if (quick) df[sample(1:nrow(df), 50000), ] else df
coordinates(spdf) = ~ lon + lat
suppressWarnings({
  slot(spdf, 'proj4string') <- CRS('+proj=longlat +ellps=sphere')
})
cells <- auto_BAUs(manifold = sphere(), type = "hex", isea3h_res = 5, data = spdf)

## Plot the cells
gg <- plot_spatial_or_ST(cells, column_names = "lat", plot_over_world = T, alpha = 0, colour = "black", size = 0.1)[[1]]
gg <- draw_world_custom(gg)
gg <- gg + theme(axis.title = element_blank(), panel.border = element_blank(), panel.background = element_blank(), legend.position = "none")
ggsv(gg, filename = "discretegrid", width = 8.5, height = 3.8, path = img_path)
ggsv(ggarrange(Zplot_norect, gg, widths = c(1.1, 1)), filename = "data_discretegrid", width = 8.5, height = 3.8, path = img_path)

# Relabel the cells from 1:length(cells)
for (i in 1:length(cells)) {
  cells@polygons[[i]]@ID <- as.character(i)
}
names(cells@polygons) <- 1:length(cells)
cells@data$id <- 1:length(cells)

# Helper function to map data to BAUs
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

x <- map_to_BAUs(spdf, sp_pols = cells)
x@data[, c("lon", "lat")] <- x@coords # true coordinate of the measurement (rather than the BAU centroids)
split_df <- group_split(x@data, id)
names(split_df) <- sapply(split_df, function(df) df$id[1])

# Only consider cells with at least 30 observations
idx <- which(sapply(split_df, function(x) nrow(x) >= 30))
split_df <- split_df[idx]

## Hexagon clusters
suppressMessages(suppressWarnings({
  sf_use_s2(FALSE) # https://github.com/r-spatial/sf/issues/1762#issuecomment-900571711
  nb <- poly2nb(cells)
}))

subset_cells <- function(cells, i, neighbour_list) {
  idx <- c(i, neighbour_list[[i]])
  bau_subset <- cells@polygons[idx]
  bau_subset <- SpatialPolygons(bau_subset, 1:length(bau_subset), cells@proj4string)
  newdata    <- data.frame(central_hexagon = as.numeric(idx == i))
  row.names(newdata) <- sapply(bau_subset@polygons, slot, "ID")
  bau_subset <- SpatialPolygonsDataFrame(bau_subset, data = newdata)
  return(bau_subset)
}

# Sanity check: plot some BAUs and their neighbours
bau_subset1 <- subset_cells(cells, 400, nb)
bau_subset2 <- subset_cells(cells, 800, nb)
bau_subset <- rbind(bau_subset1, bau_subset2)
gg <- plot_spatial_or_ST(bau_subset, "central_hexagon", plot_over_world = T)[[1]]
gg <- draw_world_custom(gg)
gg <- gg + theme(legend.position = "none", axis.title = element_blank(), panel.border = element_blank(), panel.background = element_blank())
ggsv(gg, filename = "clustering", width = 6.8, height = 3.8, path = img_path)

# Now create a list of hexagon clusters, each associated with a central hexagon.
# We just need to store the coordinates and data as a data frame.
# N == length(nb) # sanity check
clustered_data <- lapply(1:length(cells), function(i) {
  idx <- c(i, nb[[i]]) # get the neighbours of the ith BAU
  idx <- as.character(idx)
  # extract the data frames for each neighbour
  dat <- lapply(idx[idx %in% names(split_df)], function(y) split_df[[y]])
  dat <- do.call(rbind, dat)
  dat
})
names(clustered_data) <- 1:length(cells) # BAU id

# Sanity check: plot the data and the corresponding hexagon cluster to ensure that they overlap
i <- 400
tmp <- clustered_data[[i]]
tmp <- SpatialPointsDataFrame(tmp[, c("lon", "lat")], data = tmp[, "Z"])
gg1 <- plot_spatial_or_ST(tmp, "Z", plot_over_world = T)[[1]]
gg2 <- plot_spatial_or_ST(subset_cells(cells, i, nb), "central_hexagon", plot_over_world = T)[[1]]
gg3 <- ggarrange(gg1, gg2, nrow = 1)

## Remove null entries
clustered_data <- clustered_data[!sapply(clustered_data, is.null)]

# Summarise the number of observed locations, n, across the clusters:
# length(clustered_data) # 2161: number of clusters that we are making inference over
n <- sapply(clustered_data, nrow)
# summary(n)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# 30    1081    2769    3204    4976   12591

# histogram of sample sizes
gghist <- ggplot() +
  geom_histogram(aes(x = n), bins = 20, fill = "gray", colour= "black") +
  labs(x = expression("Number of observed locations,"~italic(n)), y = "Frequency") +
  theme_bw()
ggsv(gghist, filename = "samplesizes", width = 6.8, height = 3.8, path = img_path)

# Combined figure of cells and sample sizes
ggsv(ggarrange(gg, gghist, nrow = 1), filename = "clusteringsamplesizes", width = 9, height = 3, path = img_path)

## Convert from longitude-latitude to Cartesian coordinates
convert_to_cartesian <- function(lonlat, R = 6371) {
  
  lon <- lonlat[1] * pi/180
  lat <- lonlat[2] * pi/180
  
  x <- R * cos(lat) * cos(lon)
  y <- R * cos(lat) * sin(lon)
  z <- R * sin(lat)
  
  c(x, y, z)
}
clustered_data <- lapply(clustered_data, function(dat) {
  S <- as.matrix(dat[, c("lon", "lat")])
  S <- t(apply(S, 1, convert_to_cartesian))
  colnames(S) <- c("x", "y", "z")
  dat <- cbind(dat, S)
  dat[, c("lon", "lat")] <- NULL
  return(dat)
})

## Centre the data about zero
clustered_data <- lapply(clustered_data, function(dat) {
  dat$Z <- dat$Z - mean(dat$Z)
  return(dat)
})

## Scaling factors for transforming distances to [0, sqrt(2)]
scale_factor <- function(S) {
  max_dist <- max(dist(S[chull(S), ]))
  min_dist <- min(nndist(S))
  scale_factor <- sqrt(2) / max_dist
  return(scale_factor)
}
scale_factors <- sapply(clustered_data, function(dat) {
  S <- as.matrix(dat[, c("x", "y", "z")])
  scale_factor(S)
})



# ---- Save objects ----

saveRDS(clustered_data, file = file.path(int_path, "clustered_data.rds"))   # a list of data frames with data for each cell cluster
saveRDS(cells, file = file.path(int_path, "cells.rds"))                     # the cells as a "SpatialPolygonsDataFrame"
saveRDS(scale_factors, file = file.path(int_path, "scale_factors.rds"))     # scaling factors for transformation to the unit square

## Also save the list of clustered data sets as a single data frame
clustered_data <- lapply(seq_along(clustered_data), function(i) {
  dat <- clustered_data[[i]]
  dat$cluster <- i
  return(dat)
})
clustered_data2 <- do.call("rbind", clustered_data)
saveRDS(clustered_data2, file = file.path(int_path, "clustered_data2.rds"))


# ---- Unused adjacency matrix code ----

# ## Adjacency matrices
# build_dag <- function(NNarray) {
#   n = nrow(NNarray)
#   k = ncol(NNarray)
#   all_i = all_j = 1
#   for (j in 2:n) {
#     i = NNarray[j, ]
#     i = i[!is.na(i)]
#     all_j = c(all_j, rep(j, length(i)))
#     all_i = c(all_i, i)
#   }
#   sparseMatrix(i = all_i, j = all_j)
# }
# adjacency_matrix <- function(S, r = NULL, k = 10, maxmin = FALSE) {
#   
#   ## Find the neighbours
#   if (maxmin) {
#     ord <- GpGp::order_maxmin(S)
#     Sord <- S[ord, ]
#     NNarray <- GpGp::find_ordered_nn(Sord, k)
#   } else if (is.null(r)) {
#     NNarray <- knearneigh(S, k = k)$nn
#   } else {
#     # See also dpscan::frNN
#     # NB the following code breaks when there are no neighbours within a nodes 
#     # neighbourhood disc. Not fixing, because I'm now using the functionality 
#     # in the NeuralEstimators package.
#     NNarray <- nabor::knn(S, k = k, radius = r / scale_factor(S))$nn.idx
#   }
#   
#   ## Construct the adjacency matrix
#   A <- build_dag(NNarray)
#   
#   ## Add distances to A
#   ## NB inefficient to compute all of the distances when we 
#   ## only need a very small subset. This computation corresponds to about half 
#   ## of the run time, so it worthwhile improving efficiency here (particularly 
#   ## since this is quite memory inefficient)
#   indices <- which(A != 0, arr.ind = TRUE)
#   if (maxmin) {
#     D <- as.matrix(dist(Sord)) 
#   } else {
#     D <- as.matrix(dist(S))
#   }
#   d <- D[indices]
#   A <- as(A, "dMatrix")
#   A@x <- d
#   # Sanity check: A[1:6, 1:6]; D[1:6, 1:6]
#   
#   if (maxmin) {
#     # "unorder" back to the original ordering
#     # Sanity check: all(Sord[order(ord), ] == S)
#     # Sanity check: all(D[order(ord), order(ord)] == as.matrix(dist(S)))
#     A <- A[order(ord), order(ord)]
#   }
#   
#   # Remove zeros (don't want self loops)
#   A <- drop0(A)
#   
#   return(A)
# } 
# 
# adjacency_matrices <- mclapply(clustered_data, function(dat) { 
#   S <- as.matrix(dat[, c("x", "y", "z")])
#   A <- adjacency_matrix(S)
#   return(A)
# }, mc.cores = round(parallel::detectCores() / 2)
# )
# 
# ## Difficult to save R sparse matrices and load them into Julia... just save the 
# ## indices and non-zero values, and re-build sparse matrix in Julia
# convert_to_matrix <- function(A) {
#   # Find non-zeros
#   ij <- which(A != 0, arr.ind = TRUE)
#   v <- A[ij]
#   ijv <- cbind(ij, v)
#   
#   # Add diagonal elements 
#   n <- nrow(A)
#   ijv_diagonal <- matrix(c(1:n, 1:n, rep(0, n)), ncol = 3)
#   ijv <- rbind(ijv, ijv_diagonal)
#   
#   return(ijv)
# }
# adjacency_matrices <- lapply(adjacency_matrices, convert_to_matrix)
# ## Convert to single data frame (RData didn't seem to like data stored as a list)
# adjacency_matrices <- lapply(seq_along(adjacency_matrices), function(cluster) cbind(adjacency_matrices[[cluster]], cluster))
# adjacency_matrices <- do.call(rbind, adjacency_matrices)
# adjacency_matrices <- as.data.frame(adjacency_matrices)
# 
# saveRDS(adjacency_matrices, file = file.path(int_path, "adjacency_matrices.rds")) # adjacency matrices
# write.csv(adj_mat_time["elapsed"], file = file.path(int_path, "adjacency_matrices_time.csv")) # adjacency matrices time
