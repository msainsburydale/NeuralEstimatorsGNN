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
	p::Integer; d::Integer = 1, nh::Integer = 128,
	nlayers::Integer = 3, # number of layers in addition to the first layer
	propagation::String = "GraphConv",
	globalpool::String = "mean"
	)

	@assert nlayers > 0

	graphtograph = if propagation == "GraphConv"
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

	globpool = if globalpool == "mean"
		no = nh
		GlobalPool(mean)
	elseif globalpool == "deepset"
		nt = 128  # dimension of the summary vector for each node
		no = 128  # dimension of the final summary vector for each graph
		ψ = Chain(Dense(nh, nt), Dense(nt, nt))
		ϕ = Chain(Dense(nt, no), Dense(no, no))
		DeepSetPool(ψ, ϕ)
	else
		error("global pooling module not recognised")
	end

	# deepset = DeepSet(
	# 	Chain(
	# 		Dense(no => nh, relu),
	# 		Dense(nh => nh, relu),
	# 		Dense(nh => nh, relu)
	# 	),
	# 	Chain(
	# 		Dense(nh => nh, relu),
	# 		Dense(nh => nh, relu),
	# 		Dense(nh => p)
	# 	)
	# )

	deepset = DeepSet(
		identity,
		Chain(
			Dense(no => nh, relu),
			Dense(nh => nh, relu),
			Dense(nh => p)
		)
	)

	return GNN(graphtograph, globpool, deepset)
end

# # ?GNN
# p = 3
# gnn1 = gnnarchitecture(p, nh = 128, globalpool = "mean")
# nparams(gnn1)
#
# gnn2 = gnnarchitecture(p, nh = 32, globalpool = "deepset", nlayers=2)
# nparams(gnn2)
#
# d = 1
# n₁, n₂ = 200, 500                           # number of nodes
# e₁, e₂ = 30, 50                             # number of edges
# g₁ = rand_graph(n₁, e₁, ndata=rand(d, n₁))
# g₂ = rand_graph(n₂, e₂, ndata=rand(d, n₂))
# z = batch([g₁, g₂])
#
# using BenchmarkTools
# @btime gnn1(z)
# @btime gnn2(z)
