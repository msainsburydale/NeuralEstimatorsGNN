using NeuralEstimators
import NeuralEstimators: simulate
using NeuralEstimatorsGNN
using Distances: pairwise, Euclidean
using LinearAlgebra
using Folds

pts = range(0, 1, length = 16)
S = expandgrid(pts, pts)
parameter_names = String.(collect(keys(Ω)))

#TODO probably shouldn't define S here, do it in the scripts that use gridded data
ξ = (
	Ω = Ω,
	pts = pts,
	S = S,
	D = pairwise(Euclidean(), S, S, dims = 1),
	p = length(Ω),
	d = size(S, 1),
	parameter_names = parameter_names,
	ρ_idx = findfirst(parameter_names .== "ρ"),
	ν_idx = findfirst(parameter_names .== "ν"),
	ν = 1.0, # smoothness to use if ν is not included in Ω
	σ = 1.0, # marginal variance to use if σ is not included in Ω
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
