using NeuralEstimators
using NeuralEstimatorsGNN
using GraphNeuralNetworks
using Flux
using Statistics: mean

function cnnarchitecture(p, qₛ = 0)

	qₜ = 256

	ψ = Chain(
		Conv((10, 10), 1 => 64,  relu),
		Conv((5, 5),  64 => 128,  relu),
		Conv((3, 3),  128 => qₜ, relu),
		Flux.flatten
		)

	ϕ = Chain(
		Dense(qₜ + qₛ, 500, relu),
		Dense(500, p)
	)

	return ψ, ϕ
end

function dnnarchitecture(n::Integer, p::Integer)

	qₜ = 256

	ψ = Chain(
		Dense(n, 256, relu),
		Dense(256, 256, relu),
		Dense(256, qₜ, relu)
	)

	ϕ = Chain(
		Dense(qₜ, 128, relu),
		Dense(128, 64, relu),
		Dense(64, p),
		x -> dropdims(x, dims = 2)
	)

	return ψ, ϕ
end

function gnnarchitecture(
	# p::Integer;
	# d::Integer = 1,
	# nh::Integer = 128,
	# nlayers::Integer = 3, # number of propagation layers (in addition to the first layer)
	# propagation::String = "WeightedGraphConv",
	# readout::String = "mean"
	p::Integer;
	d::Integer = 1,
	nh::Integer = 64,
	nlayers::Integer = 2, # number of propagation layers (in addition to the first layer)
	propagation::String = "WeightedGraphConv",
	readout::String = "universal"
	)

	@assert nlayers > 0

	# Propagation module
	propagation = if propagation == "GraphConv"
		 GNNChain(
			GraphConv(d  => nh, relu, aggr = mean),
			[GraphConv(nh => nh, relu, aggr = mean) for _ in 1:nlayers]...
		)
	elseif propagation == "WeightedGraphConv"
		GNNChain(
			WeightedGraphConv(d  => nh, relu, aggr = mean),
			[WeightedGraphConv(nh => nh, relu, aggr = mean) for _ in 1:nlayers]...
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
		Dense(nh => p)
	)

	return GNN(propagation, readout, ϕ)
end

# ?GNN
# p = 3
# x = gnnarchitecture(p)
# x = gnnarchitecture(p, nh = 128, readout = "mean")
# y = gnnarchitecture(p, nh = 64, readout = "universal")
#
# d = 1
# n = 250                           # number of nodes
# e = 2000                          # number of edges
# g = rand_graph(n, e, ndata=rand(d, n))
# z = batch([g, g])
#
# g |> x.propagation |> x.readout
# g |> y.propagation |> y.readout
#
# using BenchmarkTools
# @btime x(z)
# @btime y(z)
