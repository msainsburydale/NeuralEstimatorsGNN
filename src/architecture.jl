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
	propagation::String = "GraphConv",
	globalpool::String = "mean"
	)

	graphtograph = if propagation == "GraphConv"
		 GNNChain(
			GraphConv(d  => nh, relu, aggr = mean),
			GraphConv(nh => nh, relu, aggr = mean),
			GraphConv(nh => nh, relu, aggr = mean),
			GraphConv(nh => nh, relu, aggr = mean)
		)
	elseif propagation == "WeightedGraphConv"
		GNNChain(
			WeightedGraphConv(d  => nh, relu, aggr = mean),
			WeightedGraphConv(nh => nh, relu, aggr = mean),
			WeightedGraphConv(nh => nh, relu, aggr = mean),
			WeightedGraphConv(nh => nh, relu, aggr = mean)
		)
	else
		error("propagation module not recognised")
	end


	globpool = if globalpool == "mean"
		no = nh
		GlobalPool(mean)
	elseif globalpool == "attention"
		no = 64  # dimension of the final summary vector for each graph
		fgate = Dense(nh, 1)
		ffeat = Dense(nh, no)
		GlobalAttentionPool(fgate, ffeat)
	elseif globalpool == "deepset"
		nt = 32  # dimension of the summary vector for each node
		no = 64  # dimension of the final summary vector for each graph
		ψ = Dense(nh, nt)
		ϕ = Dense(nt, no)
		DeepSetPool(ψ, ϕ)
	else
		error("global pooling module not recognised")
	end

	deepset = DeepSet(
		Chain(
			Dense(no => nh, relu),
			Dense(nh => nh, relu),
			Dense(nh => nh, relu)
		),
		Chain(
			Dense(nh => nh, relu),
			Dense(nh => nh, relu),
			Dense(nh => p)
		)
	)
	estimator = GNN(graphtograph, globpool, deepset)

	# ψ = PropagateReadout(graphtograph, globpool)
	# ϕ = Chain(
	# 	Dense(no => nh, relu),
	# 	Dense(nh => nh, relu),
	# 	Dense(nh => nh, relu),
	# 	Dense(nh => p)
	# )
	# estimator = DeepSet(ψ, ϕ)

	return estimator
end
