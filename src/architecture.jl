using NeuralEstimators, Flux, GraphNeuralNetworks, Statistics

function gnnarchitecture(
	p::Integer;
	nₕ = [64, 256], # number of channels in each propagation layer
	final_activation = identity
	)

	if isa(nₕ, Integer) nₕ = [nₕ] end
	nlayers = length(nₕ)               # number of propagation layers
	c = [16, fill(1, nlayers-1)...]    # number of channels for weight function w(⋅, ⋅)
	in=[1, nₕ[1:end-1]...]             # input dimensions of propagation features

	# Propagation module
	aggr = mean  # neighbourhood aggregation function
	propagation_layers = map(1:nlayers) do l
		SpatialGraphConv(in[l] => nₕ[l], relu, c=c[l], aggr = aggr)
	end
	propagation = GNNChain(propagation_layers...)

	# Readout module
	readout = GlobalPool(mean)
	nᵣ = nₕ[end] # dimension of readout vector

	# Summary network
	ψ = GNNSummary(propagation, readout)

	# Mapping module
	ϕ = Chain(Dense(nᵣ => 64, relu), Dense(64 => p, final_activation))

	return DeepSet(ψ, ϕ)
end
