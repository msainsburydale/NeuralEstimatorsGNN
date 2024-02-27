using NeuralEstimators
using Distributions
using Flux
using GraphNeuralNetworks
using Statistics: mean

function gnnarchitecture(ξ; args...)

	# Extract prior support:
	Ω = ξ.Ω
	p = length(Ω)
	a = [minimum.(values(Ω))...]
	b = [maximum.(values(Ω))...]

	# Final activation function compresses output to prior support:
	final_activation = Compress(a, b)

	return gnnarchitecture(p; final_activation = final_activation, args...)
end

function gnnarchitecture(
	p::Integer;
	propagation::String = "WeightedGraphConv",
	d::Integer = 1, 	    # dimension of the response variable (univariate by default)
	nh::Integer = 128,    # number of channels in each propagation layer
	nlayers::Integer = 4, # number of propagation layers 
	aggr = mean,          # node aggregation function
	readout::String = "mean",
	final_activation = exp
	)

	@assert nlayers >= 1

	# Propagation module
  prop = if propagation == "GraphConv"
		 GraphConv
	elseif propagation == "WeightedGraphConv"
		WeightedGraphConv
	elseif propagation == "WeightedGINConv"
		WeightedGINConv
	else
		error("propagation module not recognised")
	end
	propagation_layers = []
	push!(propagation_layers, prop(d  => nh, relu, aggr = aggr))
	if nlayers >= 2
	  push!(propagation_layers, [GraphConv(nh => nh, relu, aggr = aggr) for _ in 2:nlayers]...)
	end
	propagation = GNNChain(propagation_layers...)

	# Readout module
	readout = if readout == "mean"
		no = nh
		GlobalPool(mean)
	elseif readout == "universal"
		nt = 64  # dimension of the summary vector for each node
		no = 128  # dimension of the final summary vector for each graph
		ψ = Dense(nh, nt)
		ϕ = Dense(nt, no)
		UniversalPool(ψ, ϕ)
	else
		error("global pooling module not recognised")
	end

	# Mapping module
	ϕ = Chain(
		Dense(no => nh, relu),
		Dense(nh => nh, relu),
		Dense(nh => p, final_activation)
	)

	return GNN(propagation, readout, ϕ)
end
