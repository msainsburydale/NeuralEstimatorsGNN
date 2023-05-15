using NeuralEstimators
import NeuralEstimators: simulate
using NeuralEstimatorsGNN
using Distances: pairwise, Euclidean
using Distributions: Uniform
using LinearAlgebra
using Folds

Ω = (
	ρ = Uniform(0.05, 0.5),
	ν = Uniform(0.5, 2.0)
)
parameter_names = String.(collect(keys(Ω)))
pts = range(0, 1, length = 16)
S = expandgrid(pts, pts)
ξ = (
	Ω = Ω,
	S = S,
	D = pairwise(Euclidean(), S, S, dims = 1),
	p = length(Ω),
	parameter_names = parameter_names,
	ρ_idx = findfirst(parameter_names .== "ρ"),
	ν_idx = findfirst(parameter_names .== "ν"),
	σ = 1.0, # marginal variance to use if σ is not included in Ω
	invtransform = exp,
	ϵ = 0.125f0 # cutoff distance used to define the neighbourhood of each node
)

function simulate(parameters::Parameters, ξ, m::R) where {R <: AbstractRange{I}} where I <: Integer

	P = size(parameters, 2)
	m̃ = rand(m, P)

	chols        = parameters.chols
	chol_pointer = parameters.chol_pointer

	Z = Folds.map(1:P) do i
		L = view(chols, :, :, chol_pointer[i])
		z = simulateschlather(L, m̃[i])
		z = Float32.(z)
		z
	end
	n = size(chols, 1)
	Z = reshape.(Z, isqrt(n), isqrt(n), 1, :) # assumes a square domain
	return Z
end
simulate(parameters::Parameters, ξ, m::Integer) = simulate(parameters, ξ, range(m, m))
simulate(parameters::Parameters, ξ) = stackarrays(simulate(parameters, ξ, 1))
