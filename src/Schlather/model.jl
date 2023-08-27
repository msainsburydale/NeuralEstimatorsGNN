using NeuralEstimators
import NeuralEstimators: simulate
using NeuralEstimatorsGNN
using Distances: pairwise, Euclidean
using Distributions: Uniform
using LinearAlgebra
using Folds

Ω = (
	ρ = Uniform(0.05, 0.3),
	ν = Uniform(0.5, 1.5)
)
parameter_names = String.(collect(keys(Ω)))

ξ = (
	Ω = Ω,
	p = length(Ω),
	n = 256,
	parameter_names = parameter_names,
	ρ_idx = findfirst(parameter_names .== "ρ"),
	ν_idx = findfirst(parameter_names .== "ν"),
	σ = 1.0, # marginal variance to use if σ is not included in Ω
	δ = 0.15f0, # cutoff distance used to define the neighbourhood of each node
	invtransform = exp # inverse of variance-stabilising transformation
)

function simulate(parameters::Parameters, m::R) where {R <: AbstractRange{I}} where I <: Integer

	K = size(parameters, 2)
	m̃ = rand(m, K)

	chols        = parameters.chols
	chol_pointer = parameters.chol_pointer

	Z = Folds.map(1:K) do k
		L = chols[chol_pointer[k]][:, :]
		L = convert(Matrix, L) # FIXME shouldn't need to do this conversion. Think it's just a problem with the dispatching of simulateschlather()
		z = simulateschlather(L, m̃[k])
		z = Float32.(z)
		z
	end
	return Z
end
simulate(parameters::Parameters, m::Integer) = simulate(parameters, range(m, m))
simulate(parameters::Parameters) = stackarrays(simulate(parameters, 1))
