#TODO FRK::map_data_to_BAUs

library("dggridR")
library("dplyr")
library("NeuralEstimators")
library("JuliaConnectoR")
library("reshape2")
library("ggplot2")
library("rnaturalearth")
library("rnaturalearthdata")


#TODO
# - Figure out how to scale these cells to be contained within the unit square. 
# - Need to figure out how to calculate the distances.

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

## Drop columns 
df$sst <- df$lat2 <- NULL

# ---- Bin the data ----

dggs <- dgconstruct(res = 4)
df$cell <- dgGEO_to_SEQNUM(dggs, df$lon, df$lat)$seqnum
split_df <- group_split(df, cell)
names(split_df) <- sapply(split_df, function(df) df$cell[1])

#TODO Figure out what we want to do here
# only consider cells with at least 30 observations
# sum(table(df$cell) < 30)
# sum(table(df$cell) >= 30)
idx <- which(sapply(split_df, function(x) nrow(x) >= 10))
split_df <- split_df[idx]

# ---- Load the neural estimator ----

estimator = juliaLet('
  include(joinpath(pwd(), "src/architecture.jl"))
  estimator = gnnarchitecture(2; propagation = "WeightedGraphConv")
                      ')

estimator <- loadbestweights(estimator, "intermediates/GP/nuFixed/runs_GNN_m1")


#TODO No, the scaling has to be by a factor that is common for all grids, 
# otherwise the range parameters are not comparable. That is, need to determine 
# how to scale the given dggrid to the unit square
scale_values <- function(y){(y-min(y))/(max(y)-min(y))}


# Strategy: 
# - Split the data frame into data Z and locations S
# - Convert vectors of Z and observation locations S into a vector of GNNGraphs
# - Apply the estimator to the graph, yielding estimates over all regions


# Convert data into correct form (n x m matrix, where n is the number of 
# observations and m is the number of replicates, here equal to 1)
Z <- lapply(split_df, function(x) matrix(x$Z, nrow = 1))

# Spatial locations as matrix
S <- lapply(split_df, function(x) {
  
  # scale to [0, 1] x [0, 1]
  S <- as.matrix(x[, c("lon", "lat")]) 
  colnames(S) <- NULL
  S <- apply(S, 2, scale_values) 
  
  S
})

# Construct the graphs separately as we may need to use them again when boostrapping
g <- lapply(seq_along(S), function(i) {
  juliaLet("
           # A = adjacencymatrix(S, 0.15)
           A = adjacencymatrix(S, 2) # TODO uncomment above when finished prototyping
           GNNGraph(A)
           ", S = S[[i]])
})

# Add observed data to the graphs
Zgraph <- lapply(seq_along(g), function(i) {
  juliaLet("
           GNNGraph(g; ndata = Z)
           ", g = g[[i]], Z = Z[[i]])
})


# Apply the estimator
thetahat <- estimate(estimator, Zgraph)

# Put estimates into a data frame for plotting
colnames(thetahat) <- names(split_df)
rownames(thetahat) <- c("tau", "rho")

estimates <- melt(thetahat, varnames = c("parameter", "cell"), value.name = "estimate")

# Map the estimates to the corresponding dggrid cell
# Get the grid cell boundaries for cells which have estimates
grid <- dgcellstogrid(dggs, estimates$cell)

#Update the grid cells' properties to include the estimates of each cell
grid <- merge(grid, estimates, by.x ="seqnum", by.y="cell")


# ---- Plotting ----

#Get polygons for each country of the world
# countries <- map_data("world")
world <- ne_countries(scale = "medium", returnclass = "sf")

# Handle cells that cross 180 degrees
wrapped_grid = st_wrap_dateline(grid, options = c("WRAPDATELINE=YES","DATELINEOFFSET=180"), quiet = TRUE)

wrapped_grid <- filter(wrapped_grid, parameter == "rho")


# Plot using the "Mollweide" projection
ggplot() +
  geom_sf(data=wrapped_grid, aes(fill=estimate), color=alpha("white", 0.4)) +
  geom_sf(data=world, fill = "black") +
  coord_sf(crs = "+proj=moll +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs ") +
  # scale_fill_distiller(palette = "Spectral")
  scale_fill_distiller(palette = "BrBG") + 
  theme_bw()

