using NeuralEstimators, Flux, GraphNeuralNetworks, Statistics

function gnnarchitecture(Ω; args...)

	# Extract prior support:
	p = length(Ω)
	a = [minimum.(values(Ω))...]
	b = [maximum.(values(Ω))...]

	# Final activation function compresses output to prior support:
	final_activation = Compress(a, b)

	return gnnarchitecture(p; final_activation = final_activation, args...)
end

function gnnarchitecture(
	p::Integer;
	nₕ = [128, 128, 128, 128],    # number of channels in each propagation layer
	aggr = mean,                 # neighbourhood aggregation function
	final_activation = identity
	)

	if isa(nₕ, Integer) nₕ = [nₕ] end
	nlayers = length(nₕ)             # number of propagation layers
	c = [16, fill(1, nlayers-1)...]  # number of channels for weight function w(⋅)
	in=[1, nₕ[1:end-1]...]           # input dimensions of propagation features

	# Propagation module
	propagation_layers = map(1:nlayers) do l
		SpatialGraphConv(in[l] => nₕ[l], relu, w_scalar = true, aggr = aggr)
	end
	propagation = GNNChain(propagation_layers...)

	# Readout module
	readout = GlobalPool(mean)
	nᵣ = nₕ[end] # dimension of readout vector

	# Summary network
	ψ = GNNSummary(propagation, readout)

	# Mapping module
	ϕ = Chain(
	  Dense(nᵣ => 128, relu), 
	  Dense(nᵣ => 128, relu), 
	  Dense(128 => p, final_activation)
	  )

	return DeepSet(ψ, ϕ)
end
