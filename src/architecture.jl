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

	return architecture(p; final_activation = final_activation, args...)
end

function gnnarchitecture(
	p::Integer;
	propagation::String = "WeightedGraphConv",
	d::Integer = 1,
	nh::Integer = 128,
	nlayers::Integer = 3, # number of propagation layers (in addition to the first layer)
	readout::String = "mean",
	final_activation = exp,
	aggr = mean
	)

	@assert nlayers > 0

	# Propagation module
	propagation = if propagation == "GraphConv"
		 GNNChain(
			GraphConv(d  => nh, relu, aggr = aggr),
			[GraphConv(nh => nh, relu, aggr = aggr) for _ in 1:nlayers]...
		)
	elseif propagation == "WeightedGraphConv"
		GNNChain(
			WeightedGraphConv(d  => nh, relu, aggr = aggr),
			[WeightedGraphConv(nh => nh, relu, aggr = aggr) for _ in 1:nlayers]...
		)
	elseif propagation == "WeightedGINConv"
		GNNChain(
			WeightedGINConv(d  => nh, relu, aggr = aggr),
			[WeightedGINConv(nh => nh, relu, aggr = aggr) for _ in 1:nlayers]...
		)
	else
		error("propagation module not recognised")
	end

	# Readout module
	readout = if readout == "mean"
		no = nh
		GlobalPool(mean)
	elseif readout == "universal"
		nt = 64  # dimension of the summary vector for each node
		no = 128  # dimension of the final summary vector for each graph
		# ψ = Chain(Dense(nh, nt), Dense(nt, nt))
		# ϕ = Chain(Dense(nt, no), Dense(no, no))
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
