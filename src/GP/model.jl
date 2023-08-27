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
	δ = 0.15f0, # cutoff distance used to define the neighbourhood of each node,
	invtransform = identity # inverse of variance-stabilising transformation
)

function simulate(parameters::Parameters, m::R; convert_to_graph::Bool = true) where {R <: AbstractRange{I}} where I <: Integer

	K = size(parameters, 2)
	m̃ = rand(m, K)

	τ  			 = parameters.θ[1, :]
	chols        = parameters.chols
	chol_pointer = parameters.chol_pointer
	g            = parameters.graphs

	Z = Folds.map(1:K) do k
		L = chols[chol_pointer[k]][:, :]
		z = simulategaussianprocess(L, m̃[k])
		z = z + τ[k] * randn(size(z)...) # add measurement error
		z = Float32.(z)
		if convert_to_graph
			z = batch([GNNGraph(g[chol_pointer[k]], ndata = z[:, l, :]') for l ∈ 1:m̃[k]])
		end
		z
	end
	return Z
end
simulate(parameters::Parameters, m::Integer) = simulate(parameters, range(m, m))
simulate(parameters::Parameters) = stackarrays(simulate(parameters, 1))
