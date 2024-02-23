library("GpGp")
library("Matrix")
library("ggplot2")
library("egg")
source("src/plotting.R")

#TODO can we move this to S2neighbourhood.R

## ---- GpGp functions that will be written in Julia ----

ordermaxmin = function(locs){

  # get number of locs
  n = nrow(locs)
  k = round(sqrt(n))
  # k is number of neighbors to search over
  # get the past and future nearest neighbors
  # NNall = FNN::get.knn( locs, k = k )$nn.index
  NNall = juliaGet(getknn(t(locs), t(locs), as.integer(k)))[[1]]
  # pick a random ordering
  index_in_position = c( sample(n), rep(NA,1*n) )
  position_of_index = order(index_in_position[1:n])
  # loop over the first n/4 locations
  # move an index to the end if it is a
  # near neighbor of a previous location
  curlen = n
  nmoved = 0
  for(j in 2:(2*n) ){
    nneigh = round( min(k,n/(j-nmoved+1)) )
    cat("j=", j, "index_in_position[j] = ", index_in_position[j], "\n")
    neighbors = NNall[index_in_position[j],1:nneigh] # If missing, returns NA
    if( min( position_of_index[neighbors], na.rm = TRUE ) < j ){
      nmoved = nmoved+1
      curlen = curlen+1
      position_of_index[ index_in_position[j] ] = curlen
      index_in_position[curlen] = index_in_position[j]
      index_in_position[j] = NA
    }
  }
  ord = index_in_position[ !is.na( index_in_position ) ]

  return(ord)
}

# NB this version of ordermaxmin() is slow
rowMins = function(X) apply(X, 1, FUN = min)
ordermaxmin = function(locs) {
  D = as.matrix(dist(locs))
  vecchia.seq = as.numeric(c(which.min(D[which.min(colMeans(D))[1],]))) ## Vecchia sequence based on max-min ordering: start with most central location
  for(j in 2:n){ ## loop over locations
    vecchia.seq[j] = c(1:n)[-vecchia.seq][which.max(rowMins(matrix(D[-vecchia.seq,vecchia.seq],ncol=length(vecchia.seq))))[1]]
  }
  return(vecchia.seq)
}

library("JuliaConnectoR")
getknn = juliaEval("
          using NearestNeighbors
          function getknn(S, s, k; leafsize = 10)
            tree = KDTree(S; leafsize = leafsize)
            nn_index, nn_dist = knn(tree, s, k, true)
            nn_index = hcat(nn_index...) |> permutedims 
            nn_dist  = hcat(nn_dist...)  |> permutedims  
            nn_index, nn_dist
          end
          ")

findorderednn = function(locs,k){
  
  # number of locations
  n = nrow(locs)
  k = min(k,n-1)
  mult = 2

  # to store the nearest neighbor indices
  NNarray = matrix(NA,n,k+1)
  
  # find neighbours of first mult*k+1 locations by brute force
  maxval = min( mult*k + 1, n )
  NNarray[1:maxval,] = find_ordered_nn_brute(locs[1:maxval,,drop=FALSE],k)
  
  query_inds = min( maxval+1, n):n
  data_inds = 1:n
  
  ksearch = k
  
  while( length(query_inds) > 0 ){
    ksearch = min( max(query_inds), 2*ksearch )
    data_inds = 1:min( max(query_inds), n )
    NN = FNN::get.knnx( locs[data_inds,,drop=FALSE], locs[query_inds,,drop=FALSE], ksearch )$nn.index
    #NN = juliaGet(getknn(t(locs[data_inds,,drop=FALSE]), t(locs[query_inds,,drop=FALSE]), as.integer(ksearch)))[[1]]
    less_than_l = t(sapply( 1:nrow(NN), function(l) NN[l,] <= query_inds[l]  ))
    sum_less_than_l = apply(less_than_l,1,sum)
    ind_less_than_l = which(sum_less_than_l >= k+1)
    
    NN_k = t(sapply(ind_less_than_l,function(l) NN[l,][less_than_l[l,]][1:(k+1)] ))
    
    NNarray[ query_inds[ind_less_than_l], ] = NN_k
    
    query_inds = query_inds[-ind_less_than_l]
    
  }
  
  return(NNarray)
}

findorderednnbrute = function(locs, k){
  # find the k+1 nearest neighbors to locs[j,] in locs[1:j,]
  # by convention, this includes locs[j,], which is distance 0
  n = dim(locs)[1]
  k = min(k,n-1)
  NNarray = matrix(NA,n,k+1)
  for(j in 1:n ){
    distvec = c(fields::rdist(locs[1:j,,drop=FALSE],locs[j,,drop=FALSE]) )
    NNarray[j,1:min(k+1,j)] = order(distvec)[1:min(k+1,j)]
  }
  return(NNarray)
}


# ---- Test code ----

use_GpGp = FALSE

## Generate some locations
n1 = 16
n2 = 16
n = n1*n2
locs = as.matrix(expand.grid((1:n1)/n1, (1:n2)/n2))  
locs <- locs + rnorm(n, sd = 0.0001) # add a small amount of noise


## Calculate Vecchia ordering 
ord = if (use_GpGp) {
  GpGp::order_maxmin(locs) # NB: there is randomness here (GpGp::order_maxmin(locs) gives different results everytime, despite the locations remaining fixed), and the ordering doesn't start in the centre
} else {
  ordermaxmin(locs)        # NB: as N increases, strange patterns begin to emerge
}

## Reorder locations
locsord = locs[ord, ]

## Find ordered nearest k neighbors (parents) for each node
k = 10
NNarray = if (use_GpGp) {
  GpGp::find_ordered_nn(locsord, k)  
} else {
  findorderednn(locsord, k)  
}

system.time(GpGp::find_ordered_nn(locsord, k))
system.time(findorderednn(locsord, k))

## Build the implied dag
build_dag = function(NNarray) {
  n = nrow(NNarray)
  k = ncol(NNarray)
  
  all_i = all_j = 1
  for (j in 2:n) {
    i = NNarray[j, ]
    i = i[!is.na(i)]
    all_j = c(all_j, rep(j, length(i)))
    all_i = c(all_i, i)
  }
  R = sparseMatrix(i = all_i, j = all_j)
  return(R)
}
R = build_dag(NNarray)
# image(R)

## "Moral" version of the graph
## The moralised counterpart of a directed acyclic graph is formed by adding 
## edges between all pairs of non-adjacent nodes that have a common child, and 
## then making all edges in the graph undirected.
Q = t(R) %*% R
# image(Q) 

## number of neighbours:
apply(R, 2, sum)
apply(Q, 2, sum)



# ---- Plot illustrating the neighbourhoods ----

df <- as.data.frame(locsord)
colnames(df) <- c("x", "y")

plot_neighbours_maxmin <- function(df, N) {
  ggplot() + 
    geom_point(data = df, aes(x = x, y = y), colour = "lightgray") +
    geom_point(data = df[1:N, ], aes(x = x, y = y), colour = "black") + 
    geom_point(data = df[N, ], aes(x = x, y = y), colour = "red") +
    geom_point(data = df[NNarray[N,2:(k+1)], ], aes(x = x, y = y), colour = "orange") +
    labs(title = paste("Maxmin: location", N)) + 
    coord_fixed() + 
    theme_bw() + 
    theme(axis.title = element_blank(), panel.grid = element_blank(), plot.title = element_text(hjust = 0.5))
}

plots <- lapply(c(1, 2, 5, 50, 100, 256), function(N) plot_neighbours_maxmin(df, N))

figure_maxmin <- egg::ggarrange(plots=plots, nrow = 2)

ggsv(figure_maxmin, file = "maxminordering", width = 7.3, height = 5.6, path = "img")


# Other neighbourhood definitions
s <- as.numeric(df[1, ]) # central point
r <- 0.15
D <- as.matrix(dist(locsord))
neighbours <- which(D[1, ] < r)[-1]

plot_neighbours <- function(df, s, neighbours, r = NULL) {
  gg <- ggplot() + 
    geom_point(data = df, aes(x = x, y = y), colour = "black") + 
    geom_point(aes(x = s[1], y = s[2]), colour = "red", size = 1.5) +
    geom_point(data = df[neighbours, ], aes(x = x, y = y), colour = "orange") +
    coord_fixed() + 
    theme_bw() + 
    theme(axis.title = element_blank(), panel.grid = element_blank(), plot.title = element_text(hjust = 0.5))  
   
  if (!is.null(r)) {
    gg <- gg + annotate("path", x=s[1]+r*cos(seq(0,2*pi,len=100)), y=s[2]+r*sin(seq(0,2*pi,len=100)), colour = "orange")
  }
  
  return(gg)
}

disc <- plot_neighbours(df, s, neighbours, r) + 
  labs(title = "Disc of fixed radius")

disc_k <- plot_neighbours(df, s, sample(neighbours, k), r) + 
  labs(title = "Random-k neighbours \n within disc of fixed radius")

k_nearest <- plot_neighbours(df, s, order(D[1, ])[2:(k+1)]) + 
  labs(title = "k-nearest neighbours")

figure <- egg::ggarrange(k_nearest, disc, disc_k, nrow = 1)
ggsv(figure, file = "otherneighbourhoods", width = 8, height = 3.5, path = "img")

figure <- egg::ggarrange(plots = c(list(k_nearest, disc, disc_k), plots), nrow = 3)
ggsv(figure, file = "neighbourhoods", width = 9.4, height = 8.3, path = "img")

