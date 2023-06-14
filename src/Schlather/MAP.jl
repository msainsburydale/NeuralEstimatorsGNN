using LinearAlgebra
using NeuralEstimators
include(joinpath(pwd(), "src/MAP.jl"))

"""
	corrmatrix(D, ρ, ν)
Computes the correlation matrix (i.e., with marginal variance σ² = 1) from the
matern covariance function with range parameter `ρ` and smoothness parameter `ν`.
"""
corrmatrix(D, ρ, ν) = matern.(D, ρ, ν)


# These functions allow one to include only a subset of pairs based on spatial
# distance between sites. Setting d₀ = ∞ recovers the full pairwise likelihood.

# Note that it is slightly inefficient to pass the indices through the functions
# to subset the data at the very lowest level of logpairwiselikelihood(), since this
# means that ψ is computed for all parameter pairs, rather than just a subset.
# If I do decide to change it, I can subset the distance
# matrix and obtain the i and j indices for subsetting y as follows:
# D = D[indices]
# i = map(x -> x[1], indices)
# j = map(x -> x[2], indices)

function logpairwiselikelihood(Z::V, ψ::M, indices::I) where {T <: Number, V <: AbstractArray{T, 1}, M <: AbstractArray{T, 2}, I <: Vector{CartesianIndex{2}}}
	ll = zero(T)
	for idx ∈ indices
		i = idx[1]
		j = idx[2]
		ll += schlatherbivariatedensity(Z[i], Z[j], ψ[i, j]; logdensity = true)
	end
    return ll
end

# Indepenent replicates are stored in the second dimension of y.
function logpairwiselikelihood(Z::M, ψ::M, indices::I) where {T <: Number, M <: AbstractArray{T, 2}, I <: Vector{CartesianIndex{2}}}
	ll =  mapslices(z -> logpairwiselikelihood(z, ψ, indices), Z, dims = 1)
    return sum(ll)
end

# Wrapper to use in optimize().
function nll(θ, Z, D, Ω)

	# Constrain the estimates to be within the prior support
	θ = scaledlogistic.(θ, Ω)

	# Indices of observation pairs that are within a distance of d₀.
	# Note that the full pairwise likelihood is obtained by setting d₀ = Inf.
	indices = findall(d -> 0 < d < 0.3, triu(D)) 

	# Construct the correlation matrix from the current parameters
    ψ = corrmatrix(D, θ[1], θ[2])

	return -logpairwiselikelihood(Z, ψ, indices)
end
