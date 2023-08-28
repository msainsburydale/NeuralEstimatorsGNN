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
	δ = 0.15, # cutoff distance used to define the neighbourhood of each node,
	invtransform = identity # inverse of variance-stabilising transformation
)

function simulate(parameters::Parameters, m::R; convert_to_graph::Bool = true) where {R <: AbstractRange{I}} where I <: Integer

	K = size(parameters, 2)
	m = rand(m, K)

	τ  			 = parameters.θ[1, :]
	chols        = parameters.chols
	chol_pointer = parameters.chol_pointer
	loc_pointer  = parameters.loc_pointer
	g            = parameters.graphs

	Z = Folds.map(1:K) do k
		Lₖ = chols[chol_pointer[k]][:, :]
		mₖ = m[k]
		z = simulategaussianprocess(L, mₖ)
		z = z + τ[k] * randn(size(z)...) # add measurement error
		z = Float32.(z)
		if convert_to_graph
			gₖ = g[loc_pointer[k]]
			z = batch([GNNGraph(gₖ, ndata = z[:, l, :]') for l ∈ 1:mₖ])
		end
		z
	end
	return Z
end
simulate(parameters::Parameters, m::Integer; convert_to_graph::Bool = true) = simulate(parameters, range(m, m); convert_to_graph = convert_to_graph)
