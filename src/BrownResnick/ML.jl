using LinearAlgebra
using NeuralEstimators
include(joinpath(pwd(), "src/ML.jl"))

R"""
################################################################
### Negative log pairwise likelihood for Brown-Resnick model ###
################################################################
## par: parameter vector (range>0, smoothness in [0,2])
## data: mxS data matrix on Fréchet margins (m replicates and S locations)
## dist: distance matrix between all pairs of sites
## hmax: maximum cutoff distance for the inclusion of pairs in pairwise likelihood

dyn.load("src/BrownResnick/PairwiseLikelihoodBR.so") ## load C code

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


# Z = simulatebrownresnick(rand(n, 2), 0.5, 1.2, 30)
# nll([1.0, 1.0], Z, ξ.D, [ξ.Ω...])
# ML(Z, [1.0, 1.0], ξ.D, [ξ.Ω...])



# This is a copy of the ML function in src/ML.jl, but without parallelisation,
# since the above approach is not parallelisable.

function ML(Z::V, ξ) where {T, N, A <: AbstractArray{T, N}, V <: AbstractVector{A}}

	# Compress the data from an n-dimensional array to a matrix
	Z = flatten.(Z)

	# inverse of the variance-stabilising transform
	Z = broadcast.(ξ.invtransform, Z)

	# Since we logistic-transform the parameters during optimisation to force
	# the estimates to take reasonable values, here we provide logit-transformed
	# initial values.
	Ω = ξ.Ω
	Ω = [Ω...] # convert to array since broadcasting over dictionaries and NamedTuples is reserved
	# "Widen" the prior support so we don't get so many estimates on the boundary
	Ω = map(Ω) do x
		[minimum(x)/3, maximum(x)*3]
	end
	θ₀ = scaledlogit.(ξ.θ₀, Ω)

	# Convert to Float64 so that Cholesky factorisation doesn't throw positive
	# definite error due to rounding.
	# (When you want to broadcast broadcast, then broadcast broadcast)
	Z  = broadcast.(Float64, Z)
	θ₀ = Float64.(θ₀)

	# If Z is replicated, try to replicate θ₀ accordingly.
	K = size(θ₀, 2)
	m = length(Z)
	if m != K
		if (m ÷ K) == (m / K)
			θ₀ = repeat(θ₀, outer = (1, m ÷ K))
		else
			error("The length of the data vector, m = $m, and the number of parameter configurations, K = $K, do not match; further, m is not a multiple of K, so we cannot replicate θ to match Z.")
		end
		K = size(θ₀, 2) # update K
	end

	# Distance matrix: D may be a single matrix or a vector of matrices
	D = ξ.D
	if typeof(D) <: AbstractVector
		# If θ₀ is replicated, try to create an approporiate pointer for D based
		# on same way that chol_pointer is constructed.
		L = length(D)
		if L != K && (K ÷ L) != (K / L)
			error("The number of parameter configurations, K = $K, and the number of distances matrics, L = $L, do not match; further, K is not a multiple of L, so we cannot replicate D to match θ.")
		end
		D_pointer = repeat(1:L, inner = K ÷ L) # Note that this is exactly the same as the field chol_pointer in objects of type Parameters
	else
		D = [D]
		D_pointer = repeat([1], K)
	end
	@assert length(D_pointer) == K

	# Convert from matrix to vector of vectors
	θ₀ = [θ₀[:, k] for k ∈ 1:K]

	# Optimise
	θ̂ = map(1:K) do k
		 Dₖ = D[D_pointer[k]]
		 ML(Z[k], θ₀[k], Dₖ, Ω)
	end

	# Convert to matrix
	θ̂ = hcat(θ̂...)

	return θ̂
end
