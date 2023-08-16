using LinearAlgebra
using NeuralEstimators
include(joinpath(pwd(), "src/ML.jl"))

function covariancematrix(D; τ, ρ, ν)
	# Exploit symmetry of D to minimise the number of computations
    Σ = matern.(UpperTriangular(D), ρ, ν)
	Σ[diagind(Σ)] .+= τ^2
    return Σ
end

"""
Negative log-likelihood function to be minimised using Optim. If length(θ) > 2,
the smoothness parameter, ν, is estimated; otherwise, it is fixed to 1.
"""
function nll(θ, Z, D, Ω)

	# Constrain the estimates to be within the prior support
	θ = scaledlogistic.(θ, Ω)
	p = length(θ)
	ν = p > 2 ? θ[3] : one(eltype(θ))

	# Current covariance matrix
	Σ = covariancematrix(D, τ = θ[1], ρ = θ[2], ν = ν)

	# compute the log-likelihood function
	ℓ = gaussiandensity(Z, Σ; logdensity = true)

	return -ℓ
end
