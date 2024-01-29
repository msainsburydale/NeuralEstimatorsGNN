using LinearAlgebra
using NeuralEstimators
include(joinpath(pwd(), "src/ML.jl"))

"""
Negative log-likelihood function to be minimised using Optim.

If length(θ) > 2, the marginal standard deviation, σ, is estimated. Otherwise,
it is fixed to 1.
"""
function nll(θ, Z, D, Ω)

	ν = one(eltype(θ)) # smoothness fixed to 1

	# Constrain the estimates to be within the prior support
	θ  = scaledlogistic.(θ, Ω)
	p  = length(θ)
	σ  = p > 2 ? θ[3] : one(eltype(θ))
	σ² = σ^2

	# Current covariance matrix
	Σ = covariancematrix(D, τ = θ[1], ρ = θ[2], ν = ν, σ² = σ²)

	# compute the log-likelihood function
	ℓ = gaussiandensity(Z, Σ; logdensity = true)

	return -ℓ
end

function covariancematrix(D; τ, ρ, ν, σ²)
	# Exploit symmetry of D to minimise the number of computations
    Σ = matern.(UpperTriangular(D), ρ, ν, σ²)
	Σ[diagind(Σ)] .+= τ^2
    return Σ
end
