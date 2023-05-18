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

#TODO probably shouldn't define S here, do it in the scripts that use gridded data
pts = range(0, 1, length = 16)
S   = expandgrid(pts, pts)
parameter_names = String.(collect(keys(Ω)))

ξ = (
	Ω = Ω,
	S = S,
	D = pairwise(Euclidean(), S, S, dims = 1),
	p = length(Ω),
	parameter_names = parameter_names,
	ρ_idx = findfirst(parameter_names .== "ρ"),
	ν_idx = findfirst(parameter_names .== "ν"),
	σ = 1.0, # marginal variance to use if σ is not included in Ω
	ϵ = 0.125f0, # cutoff distance used to define the neighbourhood of each node
	invtransform = exp #TODO need to incorporate this in the pairwise MAP if it's not already
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
simulate(parameters::Parameters, ξ, m::Integer) = simulate(parameters, ξ, range(m, m))
simulate(parameters::Parameters, ξ) = stackarrays(simulate(parameters, ξ, 1))
