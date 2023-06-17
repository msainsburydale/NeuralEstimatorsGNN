using LinearAlgebra
using NeuralEstimators
include(joinpath(pwd(), "src/MAP.jl"))

R"""
################################################################
### Negative log pairwise likelihood for Brown-Resnick model ###
################################################################
## par: parameter vector (range>0, smoothness in [0,2])
## data: mxS data matrix on Fréchet margins (m replicates and S locations)
## dist: distance matrix between all pairs of sites
## hmax: maximum cutoff distance for the inclusion of pairs in pairwise likelihood

dyn.load("src/BrownResnick/BrownResnickC/PairwiseLikelihoodBR.so") ## load C code

NegLogPairwiseLikelihoodBR <- function(par, data, distmat, hmax) {
  range  <- par[1]
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
"""

function nll(θ, Z, D, Ω)

	# Constrain the estimates to be within the prior support
	θ = scaledlogistic.(θ, Ω)

	par     = θ
	data    = permutedims(Z)
	distmat = D
	hmax    = 0.3

	@rput par
	@rput data
	@rput distmat
	@rput hmax

	R"""
	negloglik = NegLogPairwiseLikelihoodBR(par=par, data=data, distmat=distmat, hmax=hmax)
	"""

	@rget negloglik

	return negloglik
end


# Z = simulatebrownresnick(ξ.S, 0.5, 1.2, 30)
# nll([1.0, 1.0], Z, ξ.D, [ξ.Ω...])
# MAP(Z, [1.0, 1.0], ξ.D, [ξ.Ω...])
