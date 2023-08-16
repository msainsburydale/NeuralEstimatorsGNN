using LinearAlgebra
using NeuralEstimators
include(joinpath(pwd(), "src/ML.jl"))

corrmatrix(D, ρ, ν) = matern.(D, ρ, ν)

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
