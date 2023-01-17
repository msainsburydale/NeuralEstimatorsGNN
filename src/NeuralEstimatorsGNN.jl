module NeuralEstimatorsGNN

using LinearAlgebra: diagind
using Flux: @functor, glorot_uniform
using GraphNeuralNetworks
using GraphNeuralNetworks: check_num_nodes
using Random: seed!
export seed!

export WeightedGraphConv
export DeepSetPool
export adjacencymatrix
export reshapedataDNN, reshapedataGNN



# ---- WeightedGraphConv ----

#TODO change documentation
@doc raw"""
    WeightedGraphConv(in => out, σ=identity; aggr=+, bias=true, init=glorot_uniform)
Graph convolution layer from Reference: [Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks](https://arxiv.org/abs/1810.02244).
Performs:
```math
\mathbf{x}_i' = W_1 \mathbf{x}_i + \square_{j \in \mathcal{N}(i)} W_2 \mathbf{x}_j
```
where the aggregation type is selected by `aggr`.
# Arguments
- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `σ`: Activation function.
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).
- `bias`: Add learnable bias.
- `init`: Weights' initializer.
"""
struct WeightedGraphConv{W<:AbstractMatrix,B,F,A, C} <: GNNLayer
    W1::W
    W2::W
    W3::C
    bias::B
    σ::F
    aggr::A
end

@functor WeightedGraphConv

function WeightedGraphConv(ch::Pair{Int,Int}, σ=identity; aggr=+,
                   init=glorot_uniform, bias::Bool=true)
    in, out = ch
    W1 = init(out, in)
    W2 = init(out, in)
    # NB Even though W3 is a scalar, it needs to be stored as an array so that
    # it is recognised as a trainable field. Note that we could have a different
    # range parameter for each channel, in which case W3 would be an array of parameters.
    W3 = init(1)
    b = bias ? Flux.create_bias(W1, true, out) : false
    WeightedGraphConv(W1, W2, W3, b, σ, aggr)
end

rangeparameter(l::WeightedGraphConv) = exp.(l.W3)

function (l::WeightedGraphConv)(g::GNNGraph, x::AbstractMatrix)
    check_num_nodes(g, x)
    r = rangeparameter(l)  # strictly positive range parameter
    d = g.graph[3]         # vector of spatial distances
    w = exp.(-d ./ r)       # weights defined by exponentially decaying function of distance
    m = propagate(w_mul_xj, g, l.aggr, xj=x, e=w)
    x = l.σ.(l.W1 * x .+ l.W2 * m .+ l.bias)
    return x
end

function Base.show(io::IO, l::WeightedGraphConv)
    in_channel  = size(l.W1, ndims(l.W1))
    out_channel = size(l.W1, ndims(l.W1)-1)
    print(io, "WeightedGraphConv(", in_channel, " => ", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end


# ---- Deep Set pooling layer ----

#TODO I think there is an earlier paper that does this too. Or, perhaps the
# universal pooling that I encountered previously applied the Deep Set
# architecture over each graph (that is how I originally thought that it would
# be implemented).
@doc raw"""
    DeepSetPool(ψ, ϕ = identity)
Deep Set readout layer from the [Graph Neural Networks with Adaptive Readouts](https://arxiv.org/abs/2211.04952) paper.
It takes the form,
```math
``\mathbf{h}_V`` = ϕ(|V|⁻¹ \sum_{v\in V} ψ(\mathbf{h}_v)),
```
where ``\mathbf{h}_V`` denotes the summary vector for graph ``V``,
``\mathbf{h}_v`` denotes the vector of hidden features for node ``v \in V``,
and ψ and ψ are neural networks.

# Examples
```julia
using Graphs: random_regular_graph

# Create the pooling layer
nh = 16  # number of channels in the feature graph output from the propagation module
nt = 32  # dimension of the summary vector for each node
no = 64  # dimension of the final summary vector for each graph
ψ = Dense(nh, nt)
ϕ = Dense(nt, no)
dspool = DeepSetPool(ψ, ϕ)

# Toy input graph containing subgraphs
num_nodes  = 10
num_edges  = 4
num_graphs = 3
h = GNNGraph(random_regular_graph(num_nodes, num_edges), ndata = rand(nh, num_nodes))
h = Flux.batch([h, h, h])

# Apply the pooling layer
dspool(h)
```
"""
struct DeepSetPool{G,F}
    ψ::G
    ϕ::F
end

@functor DeepSetPool

DeepSetPool(ψ) = DeepSetPool(ϕ, identity)

function (l::DeepSetPool)(g::GNNGraph, x::AbstractArray)
    u = reduce_nodes(mean, g, l.ψ(x))
    t = l.ϕ(u)
    return t
end

(l::DeepSetPool)(g::GNNGraph) = GNNGraph(g, gdata = l(g, node_features(g)))




# ---- Misc. ----

function adjacencymatrix(D::M; ϵ, weighted = true, remove_diagonals = true) where M <: AbstractMatrix{T} where T
    A = D .< ϵ
	remove_diagonals && A[diagind(A)] .= 0 # remove the diagonal entries

	if weighted
		indices = copy(A)
		A = convert(Matrix{T}, A)
		A[indices] = D[indices]
	end

	return A
end


# ---- Reshaping data to the correct form ----



function reshapedataDNN(Z)
	reshape.(Z, :, 1, size(Z[1])[end])
end

function reshapedataGNN(Z, g::GNNGraph)
	map(Z) do z
		m = size(z)[end]
		colons  = ntuple(_ -> (:), ndims(z) - 1)
		v = [GNNGraph(g, ndata = (Matrix(vec(z[colons..., i])'))) for i ∈ 1:m]
		Flux.batch(v)
	end
end

function reshapedataGNN(Z, v::V) where {V <: AbstractVector{A}} where A
	@assert length(Z) == length(v)
	l = length(Z)
	map(1:l) do j
		z = Z[j]
		m = size(z)[end]
		colons  = ntuple(_ -> (:), ndims(z) - 1)
		Flux.batch([GNNGraph(v[j], ndata = (Matrix(vec(z[colons..., i])'))) for i ∈ 1:m])
	end
end



end
