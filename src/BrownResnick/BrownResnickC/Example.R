######################################
### Load packages and source files ###
######################################

rm(list=ls())

wd <- "~/Dropbox/ForMatt"
setwd(wd)

source('simulation_Dombry_et_al.R',chdir=TRUE)
source('FitBR.R')
dyn.load("PairwiseLikelihoodBR.so") ## load C code


########################################################
### Simulation of Brown-Resnick max-stable processes ###
########################################################

m <- 500 ## sample size (i.e., number of independent replicates)
n <- 50 ## number of spatial locations
loc <- cbind(runif(n),runif(n)) ## sampling locations uniformly in unit square [0,1]^2

par.true <- c(0.5,1) ## true parameter values (here, range=0.5, and smooth=1)

vario <- function(x){ ## power variogram (h/range)^smooth
  range <- par.true[1]
  smooth <- par.true[2]
  variogram <- (sqrt(sum(x^2))/range)^smooth
  return(variogram)
}

data <- simu_extrfcts(model="brownresnick",no.simu=m,coord=loc,vario=vario)$res ## data simulation
qqplot(data[,1],-1/log(c(1:m)/(m+1)),log="xy"); abline(0,1,col="red") ## verify that margins are unit Fréchet

#############################################################
### Fitting of Brown-Resnick model by pairwise likelihood ###
#############################################################

par.init <- c(3,0.5) ## Initial parameters
hmax <- 0.3 ## Cutoff distance for the inclusion of pairs in the pairwise likelihood

par.est <- FitBR(data,loc,hmax=hmax,par.init=par.init,hessian=FALSE)$par ## fit Brown-Resnick model by pairwise likelihood

compare <- rbind(par.true,par.est)
colnames(compare) <- c("range","smooth")
rownames(compare) <- c("true","estimate")
compare
