using LinearAlgebra
using NeuralEstimators
include(joinpath(pwd(), "src/ML.jl"))

function covariancematrix(D; σₑ, ρ, ν)
    Σ = matern.(D, ρ, ν)
	Σ[diagind(Σ)] .+= σₑ^2
    return Σ
end

"""
Negative log-likelihood function to be minimised using Optim. If length(θ) > 2,
the smoothness parameter, ν, is estimated; otherwise, it is fixed to 1.
"""
function nll(θ, Z, ξ, Ω)

	# Constrain the estimates to be within the prior support
	θ = scaledlogistic.(θ, Ω)
	ν = length(θ) > 2 ? θ[3] : 1

	# compute the log-likelihood function
	ℓ = ll(θ, ν, Z, ξ.D)

	return -ℓ
end

# This method can be used for Z <: AbstractVector and for Z <: AbstractMatrix
function ll(θ, ν, Z, D)
	Σ = covariancematrix(D, σₑ = θ[1], ρ = θ[2], ν = ν)
	ℓ = gaussiandensity(Z, Σ; logdensity = true)
	return ℓ
end

function ll(θ, ν, Z::A, D) where {T, V <: AbstractVector{T}, A <: AbstractVector{V}}
	# The spatial locations may vary between replicates so D is assumed to be an
	# array of matrices.
	m = length(Z)
	ℓ = [ll(θ, ν, vec(Z[i]), ξ.D[i]) for i ∈ 1:m]
	ℓ = sum(ℓ)
	return ℓ
end
