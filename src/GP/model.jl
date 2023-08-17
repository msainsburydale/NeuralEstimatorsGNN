using NeuralEstimators
import NeuralEstimators: simulate
using NeuralEstimatorsGNN
using Distances: pairwise, Euclidean
using LinearAlgebra
using Folds

ξ = (
	Ω = Ω,
	p = length(Ω),
	n = 256,
	parameter_names = String.(collect(keys(Ω))),
	ν = 1.0, # smoothness to use if ν is not included in Ω
	σ = 1.0, # marginal standard deviation to use if σ is not included in Ω
	r = 0.15f0, # cutoff distance used to define the neighbourhood of each node,
	invtransform = identity # inverse of variabce-stabilising transformation
)

function simulate(parameters::Parameters, m::R) where {R <: AbstractRange{I}} where I <: Integer

	K = size(parameters, 2)
	m̃ = rand(m, K)

	τ  = parameters.θ[1, :]
	chols        = parameters.chols
	chol_pointer = parameters.chol_pointer

	Z = Folds.map(1:K) do i
		L = chols[chol_pointer[i]][:, :]
		z = simulategaussianprocess(L, m̃[i])
		z = z + τ[i] * randn(size(z)...) # add measurement error
		z = Float32.(z)
		z
	end
	return Z
end
simulate(parameters::Parameters, m::Integer) = simulate(parameters, range(m, m))
simulate(parameters::Parameters) = stackarrays(simulate(parameters, 1))
