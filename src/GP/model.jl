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
	ν = 1.0,  # smoothness to use if ν is not included in Ω
	σ = 1.0,  # marginal standard deviation to use if σ is not included in Ω
	r = 0.10, # cutoff distance used to define the neighbourhood of each node
	k = 10,   # maximum number of neighbours to consider when constructing the neighbourhood
	neighbourhood = "maxmin", # neighbourhood definition
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

	z = Folds.map(1:K) do k
		Lₖ = chols[chol_pointer[k]][:, :]
		τₖ = τ[k]
		mₖ = m[k]
		zₖ = simulategaussianprocess(Lₖ, mₖ)
		zₖ = zₖ + τₖ * randn(size(zₖ)...) # add measurement error
		zₖ = Float32.(zₖ)
		if convert_to_graph
			gₖ = g[loc_pointer[k]]
			zₖ = spatialgraph(gₖ, zₖ)
		end
		zₖ
	end

	return z
end
simulate(parameters::Parameters, m::Integer; kwargs...) = simulate(parameters, range(m, m); kwargs...)
