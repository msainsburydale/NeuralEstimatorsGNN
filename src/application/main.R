library("NeuralEstimators")
library("JuliaConnectoR")
library("reshape2")
library("ggplot2")
library("dggrids")
library("dplyr")
library("FRK")
library("sp")

# library("geosphere")

img_path <- "img/application/SST"
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)


# ---- Binning function ----




map_to_BAUs <- function(data_sp, sp_pols) {
  
  ## Suppress bindings warnings
  . <- BAU_name <- NULL
  
  ## Add BAU ID to the data frame of the SP object
  sp_pols$BAU_name <- as.character(row.names(sp_pols))
  
  ## Add coordinates to @data if not already there
  if(!(all(coordnames(sp_pols) %in% names(sp_pols@data))))
    sp_pols@data <- cbind(sp_pols@data,coordinates(sp_pols))
  
  ## Find which fields in the data object are not already declared in the BAUs
  ## These are the variables we will average over
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
  data_over_sp <- cbind(data_df,data_over_sp)
  
  if(any(is.na(data_over_sp$BAU_name))) {  # data points at 180 boundary or outside BAUs -- remove
    ii <- which(is.na((data_over_sp$BAU_name)))
    data_sp <- data_sp[-ii,]
    data_over_sp <- data_over_sp[-ii,]
    warning("Removing data points that do not fall into any BAUs.
                              If you have simulated data, please ensure no simulated data fall on a
                              BAU boundary as these classify as not belonging to any BAU.")
  }
  
  new_sp_pts <- SpatialPointsDataFrame(
    coords=data_sp@coords,         # coordinates of summarised data
    data= data_over_sp ,                                # data frame
    proj4string = CRS(FRK:::.rawproj4string(data_sp)))     # CRS of original data
  
  new_sp_pts
}


# ---- Custom draw_world ----

draw_world_custom <- function(g = ggplot() + theme_bw() + xlab("") + ylab(""),inc_border = TRUE) {
  
  ## Basic checks
  if(!(is(g, "ggplot"))) stop("g has to be of class ggplot")
  if(!(is.logical(inc_border))) stop("inc_border needs to be TRUE or FALSE")
  
  ## Suppress bindings warning
  long <- lat <- group <- NULL
  
  ## Load the world map data from the FRK package
  data(worldmap, envir=environment(), package = "FRK")
  
  ## Homogenise (see details) to avoid lines crossing the map
  worldmap <- FRK:::.homogenise_maps(worldmap)
  
  
  ## Custom code 
  
  
  ## Now return a gg object with the map overlayed
  g + geom_polygon(data = worldmap, aes(x=long, y=lat, group=group), fill="black", size=0.1)
}



# ---- Load the data ----

load("data/SST/SST_sub_1000000.rda")
df <- SST_sub_1000000
df$error <- df$bias  <- NULL # remove columns that will not be used

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


# ---- Bin the data ----

coordinates(df) = ~ lon + lat
slot(df, 'proj4string') <- CRS('+proj=longlat +ellps=sphere')
baus <- auto_BAUs(manifold = sphere(), type = "hex", isea3h_res = 5, data = df)
x <- map_to_BAUs(df, sp_pols = baus)
x@coords[1:6,]               # true coordinate of the measurement
x@data[1:6, c("lon", "lat")] # coordinate of the BAU centroid
x@data[, c("lon", "lat")] <- x@coords
split_df <- group_split(x@data, id)
names(split_df) <- sapply(split_df, function(df) df$id[1])

# Only consider cells with at least 30 observations
# sum(table(df$cell) < 30)
# sum(table(df$cell) >= 30)
idx <- which(sapply(split_df, function(x) nrow(x) >= 10))
split_df <- split_df[idx]


# ---- Plot the raw data ----

## A palette for the data and predictions
nasa_palette <- c(
  "#03006d","#02008f","#0000b6","#0001ef","#0000f6","#0428f6","#0b53f7",
           "#0f81f3","#18b1f5","#1ff0f7","#27fada","#3efaa3","#5dfc7b","#85fd4e",
           "#aefc2a","#e9fc0d","#f6da0c","#f5a009","#f6780a","#f34a09","#f2210a",
           "#f50008","#d90009","#a80109","#730005"
)

Zplot <- plot_spatial_or_ST(df, column_names = "Z", plot_over_world = T, pch = 46)[[1]]
Zplot <- draw_world_custom(Zplot)
Zplot <- Zplot + 
  scale_colour_gradientn(colours = nasa_palette) + 
  labs(colour = expression(Z~(degree*C)), x = "longitude (deg)", y = "latitude (deg)") + 
  theme(legend.title.align = 0.5)

ggsave( 
  Zplot,
  filename = "data.png", device = "png", width = 6, height = 2.8,
  path = img_path
)


# ---- Load the neural estimator ----

estimator = juliaLet('
  include(joinpath(pwd(), "src/architecture.jl"))
  estimator = gnnarchitecture(2; propagation = "WeightedGraphConv")
                      ')

estimator <- loadbestweights(estimator, "intermediates/GP/nuFixed/runs_GNN_m1")

# ----

#TODO the scaling has to be by a factor that is common for all grids, 
# otherwise the range parameters are not comparable. That is, need to determine 
# how to scale each bau to the unit square
# scale_values <- function(y){(y-min(y))/(max(y)-min(y))}
# 
# #TODO base the scaling on chord length
# # chord_length <-  function(lon1, lat1, lon2, lat2) {
# #   a <- sin((lat2 - lat1)/2)^2 + cos(lat1) + cos(lat2) + sin((lon2 - lon1)/2)^2
# #   2 * sqrt(a)
# # }
# 
# # Based on: https://www.movable-type.co.uk/scripts/latlong.html
# chord_length <-  function(lonlat1, lonlat2) {
#   
#   lon1 <- lonlat1[1] * pi/180
#   lat1 <- lonlat1[2] * pi/180
#   lon2 <- lonlat2[1] * pi/180
#   lat2 <- lonlat2[2] * pi/180
#   
#   dlat <- lat2 - lat1
#   dlon <- lon2 - lon1
#   
#   a <- sin(dlat/2)^2 + cos(lat1) * cos(lat2) * sin(dlon/2)^2
#   2 * sqrt(a)
#   
# }
# 
# chord_length_matrix <- function(s){ 
#   apply(s, 1, function(x) {
#     apply(s, 1, function(y) {
#       chord_length(x, y)
#     })
#   })
# }
# 
# 
# # Scale the distances so that they are between 0 and sqrt(2)
# scaled_chord_length_matrix <- function(s) {
#   D <- chord_length_matrix(s)
#   sqrt(2) * scale_values(D)
# }
# 
# scaled_chord_length_matrix(s)


# # Spatial locations as matrix
# S <- lapply(split_df, function(x) {
#   
#   # scale to [0, 1] x [0, 1]
#   S <- as.matrix(x[, c("lon", "lat")]) 
#   colnames(S) <- NULL
#   S <- apply(S, 2, scale_values) 
#   
#   S
# })

# ---- Attempt 2 ----


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

# Convert data into correct form (n x m matrix, where n is the number of 
# observations and m is the number of replicates, here equal to 1)
Z <- lapply(split_df, function(x) matrix(x$Z, nrow = 1))
Z <- lapply(Z, function(z) z - mean(z)) # centre the data around zero


# Spatial distance matrix
S <- lapply(split_df, function(x) {
  
  # scale to [0, 1] x [0, 1]
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

# Apply the estimator
thetahat <- estimate(estimator, Zgraph)


# inverse of scale transformation to range parameter
thetahat[2, ] <- thetahat[2, ] / scales


# ---- Plot the estimates 

# Put estimates into a data frame 
colnames(thetahat) <- names(split_df)
rownames(thetahat) <- c("tau", "rho")
estimates <- melt(thetahat, varnames = c("parameter", "id"), value.name = "estimate")

# merge estimates into bau object
rho <- merge(baus, filter(estimates, parameter == "rho"))
rho_plot <- plot_spatial_or_ST(rho, column_names = "estimate", plot_over_world = T)[[1]]
rho_plot <- draw_world_custom(rho_plot)
rho_plot <-
  rho_plot + 
  # scale_fill_gradientn(colours = nasa_palette, na.value = NA) +
  scale_fill_distiller(palette = "YlOrRd", na.value = NA) +
  labs(fill = expression(hat(rho)~(km)), x = "longitude (deg)", y = "latitude (deg)") 

ggsave( 
  rho_plot,
  filename = "estimates_rho.png", device = "png", width = 6, height = 2.8,
  path = img_path
)


# ---- Plot the estimates 

# merge estimates into bau object
tau <- merge(baus, filter(estimates, parameter == "tau"))
tau@data$logestimate <- log(tau@data$estimate)
tau_plot <- plot_spatial_or_ST(tau, column_names = "logestimate", plot_over_world = T)[[1]]
tau_plot <- draw_world_custom(tau_plot)
tau_plot <- tau_plot + 
  # scale_fill_gradientn(colours = nasa_palette, na.value = NA) + 
  scale_fill_distiller(palette = "YlOrRd", na.value = NA) +
  labs(fill = expression(log(hat(tau))), x = "longitude (deg)", y = "latitude (deg)") 

ggsave( 
  tau_plot,
  filename = "estimates_tau.png", device = "png", width = 6, height = 2.8,
  path = img_path
)


# ---- Combined plots ----


library("ggpubr")

ggsave( 
  ggpubr::ggarrange(Zplot, rho_plot, tau_plot, align = "hv", nrow = 1, legend = "top"),
  filename = "all.png", device = "png", width = 10, height = 4,
  path = img_path
)


