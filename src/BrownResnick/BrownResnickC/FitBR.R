################################################################
### Negative log pairwise likelihood for Brown-Resnick model ###
################################################################
## par: parameter vector (range>0, smoothness in [0,2])
## data: mxS data matrix on Fréchet margins (m replicates and S locations)
## dist: distance matrix between all pairs of sites
## hmax: maximum cutoff distance for the inclusion of pairs in pairwise likelihood

NegLogPairwiseLikelihoodBR <- function(par, data, distmat, hmax) {
  range <- par[1]
  smooth <- par[2]
  
  if (range <= 0 | smooth <= 0 | smooth > 2) {
    return(Inf)
  } else {
    S <- ncol(data)
    n <- nrow(data)
    data2 <- data
    data2[is.na(data2)] <- -1000
    
    res <- .C("LogPairwiseLikelihoodBR", range = as.double(range), smooth = as.double(smooth), obs = as.double(t(data2)), distmat = as.double(distmat), S = as.integer(S), n = as.integer(n), hmax = as.double(hmax), output = as.double(0))$output
    
    return(-res)
  }
}

FitBR <- function(data,loc,hmax,par.init,hessian=FALSE){
  distmat <- as.matrix(dist(loc))
  fit <- optim(par = par.init, fn = NegLogPairwiseLikelihoodBR, data = data, distmat = distmat, hmax=hmax, method = "Nelder-Mead", control = list(maxit = 10000), hessian=hessian)  
  return(fit)
}
