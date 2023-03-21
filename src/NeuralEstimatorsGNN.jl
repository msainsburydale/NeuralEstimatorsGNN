module NeuralEstimatorsGNN

using NeuralEstimators
using LinearAlgebra: diagind
using Flux
using Flux: @functor, glorot_uniform
using GraphNeuralNetworks
using GraphNeuralNetworks: check_num_nodes
using Random: seed!
using Statistics: mean
export seed!

export WeightedGraphConv
export DeepSetPool
export adjacencymatrix
export reshapedataDNN, reshapedataGNN
export Parameters



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
struct WeightedGraphConv{W<:AbstractMatrix,B,F,A,C} <: GNNLayer
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




# ---- Adjacency matrices ----

# See https://en.wikipedia.org/wiki/Heap_(data_structure) for a description
# of the heap data structure, and see
# https://juliacollections.github.io/DataStructures.jl/latest/heaps/
# for a description of Julia's implementation of the heap data structure.

using SparseArrays
using LinearAlgebra
using Distances

#TODO could easily parallelise this to speed things up

"""
	adjacencymatrix(M::Matrix, k::Integer)
	adjacencymatrix(M::Matrix, ϵ::Float)

Computes a spatially weighted adjacency matrix from `M` based on either the `k`
nearest neighbours to each location, or a spatial radius of `ϵ` units.

If `M` is a square matrix, is it treated as a distance matrix; otherwise, it
should be an n x d matrix, where n is the number of spatial locations and d is
the spatial dimension (typically d = 2).

# Examples
```
using NeuralEstimatorsGNN
using Distances

n = 10
d = 2
S = rand(n, d)
k = 5
ϵ = 0.3

# Memory efficient constructors (avoids constructing the full distance matrix D)
adjacencymatrix(S, k)
adjacencymatrix(S, ϵ)

# Construct from full distance matrix D
D = pairwise(Euclidean(), S, S, dims = 1)
adjacencymatrix(D, k)
adjacencymatrix(D, ϵ)
```
"""
function adjacencymatrix(M::Mat, k::Integer) where Mat <: AbstractMatrix{T} where T

	I = Int64[]
	J = Int64[]
	V = Float64[]
	n = size(M, 1)
	m = size(M, 2)

	for i ∈ 1:n

		if m == n
			# since we have a square matrix, it's reasonable to assume that S
			# is actually a distance matrix, D:
			d = M[i, :]
		else
			# Compute distances between sᵢ and all other locations
			d = colwise(Euclidean(), M', M[i, :])
		end

		# Replace d(s) with Inf so that it's not included in the adjacency matrix
		d[i] = Inf

		# Find the neighbours of s
		j, v = findneighbours(d, k)

		push!(I, repeat([i], inner = k)...)
		push!(J, j...)
		push!(V, v...)
	end

	return sparse(I,J,V,n,n)
end

function adjacencymatrix(M::Mat, ϵ::F) where Mat <: AbstractMatrix{T} where {T, F <: AbstractFloat}

	@assert ϵ > 0

	n = size(M, 1)
	m = size(M, 2)

	if m == n

		D = M
		# bit-matrix specifying which locations are ϵ-neighbours
		A = D .< ϵ
		A[diagind(A)] .= 0 # remove the diagonal entries

		# replace non-zero elements of A with the corresponding distance in D
		indices = copy(A)
		A = convert(Matrix{T}, A)
		A[indices] = D[indices]

		# convert to sparse matrix
		A = sparse(A)
	else

		S = M

		I = Int64[]
		J = Int64[]
		V = Float64[]
		for i ∈ 1:n

			# Compute distances between s and all other locations
			s = S[i, :]
			d = colwise(Euclidean(), S', s)

			# Replace d(s) with Inf so that it's not included in the adjacency matrix
			d[i] = Inf

			# Find the ϵ-neighbours of s
			j = d .< ϵ
			j = findall(j)

			push!(I, repeat([i], inner = length(j))...)
			push!(J, j...)
			push!(V, d[j]...)
		end
		A = sparse(I,J,V,n,n)
	end

	return A
end


function findneighbours(d, k::Integer)
	V = partialsort(d, 1:k)
	J = [findfirst(v .== d) for v ∈ V]
    return J, V
end

# NB investigate why I can't get this to work when I have more time (it's very
# close). I think this approach will be more efficient than the above method.
# Approach using the heap data structure (can't get it to work properly, for some reason)
#using DataStructures # heap data structure
# function findneighbours(d, k::Integer)
#
# 	@assert length(d) > k
#
#     # Build a max heap of differences with first k elements
# 	h = MutableBinaryMaxHeap(d[1:k])
#
#     # For every element starting from (k+1)-th element,
#     for j ∈ (k+1):lastindex(d)
#         # if the difference is less than the root of the heap, replace the root
#         if d[j] < first(h)
#             pop!(h)
#             push!(h, d[j])
#         end
#     end
#
# 	# Extract the indices with respect to d and the corresponding distances
# 	J = broadcast(x -> x.handle, h.nodes)
# 	V = broadcast(x -> x.value, h.nodes)
#
# 	# # Sort by the index of the original vector d (this ordering may be necessary for constructing sparse arrays)
# 	# perm = sortperm(J)
# 	# J = J[perm]
# 	# V = V[perm]
#
# 	perm = sortperm(V)
# 	J = J[perm]
# 	V = V[perm]
#
#     return J, V
# end






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



# ---- Parameters definitions and constructors ----

# See here: https://discourse.julialang.org/t/filtering-keys-out-of-named-tuples/73564/8
drop(nt::NamedTuple, key::Symbol) =  Base.structdiff(nt, NamedTuple{(key,)})
drop(nt::NamedTuple, keys::NTuple{N,Symbol}) where {N} = Base.structdiff(nt, NamedTuple{keys})

# This is concretely typed so that simulate(params::Parameters, ξ, m::R) is
# type stable. Note that chol_idx[i] gives the Cholesky factor associated
# with parameter configuration θ[:, i].
struct Parameters{T, I} <: ParameterConfigurations
	θ::Matrix{T}
	chols::Array{Float64, 3}
	chol_idx::Vector{I}
end


function Parameters(ξ, K::Integer; J::Integer = 1)

	# All parameters not associated with the Gaussian process
	θ = [rand(ϑ, K * J) for ϑ in drop(ξ.Ω, (:ρ, :ν, :σ))]

	# Determine if we are estimating ν and σ
	estimate_ν = "ν" ∈ ξ.parameter_names
	estimate_σ = "σ" ∈ ξ.parameter_names

	# Covariance parameters associated with the Gaussian process
	ρ = rand(ξ.Ω.ρ, K)
	ν = estimate_ν ? rand(ξ.Ω.ν, K) : fill([ξ.ν], K)
	σ = estimate_σ ? rand(ξ.Ω.σ, K) : fill([ξ.σ], K)
	chols = maternchols(ξ.D, ρ, ν, σ.^2)
	ρ = repeat(ρ, inner = J)
	ν = repeat(ν, inner = J)
	σ = repeat(σ, inner = J)

	# Now insert ρ (and possibly ν and σ) into θ.
	θ₁ = θ[1:(ξ.ρ_idx-1)]
	θ₂ = θ[ξ.ρ_idx:end] # Note that ρ and ν are not in θ, so we don't need to skip any indices.
	if estimate_ν && estimate_σ
		@assert (ξ.ν_idx == ξ.ρ_idx + 1) && (ξ.σ_idx == ξ.ν_idx + 1) "The code assumes that ρ, ν, and σ are stored continguously in θ and in that order, that is, that (ξ.ν_idx == ξ.ρ_idx + 1) && (ξ.σ_idx == ξ.ν_idx + 1)"
		θ = [θ₁..., ρ, ν, σ, θ₂...]
	elseif estimate_ν
		@assert ξ.ν_idx == ξ.ρ_idx + 1 "The code assumes that ρ and ν are stored continguously in θ, that is, that ξ.ν_idx == ξ.ρ_idx + 1"
		θ = [θ₁..., ρ, ν, θ₂...]
	else
		θ = [θ₁..., ρ, θ₂...]
	end

	# Concatenate into a matrix and convert to Float32 for efficiency
	θ = hcat(θ...)'
	θ = Float32.(θ)

	Parameters(θ, chols, objectindices(chols, θ))
end

"""
	objectindices(objects, θ::AbstractMatrix{T}) where T
Returns a vector of indices giving element of `objects` associated with each
parameter configuration in `θ`.

The number of parameter configurations, `K = size(θ, 2)`, must be a multiple of
the number of objects, `N = size(objects)[end]`. Further, repeated parameters
used to generate `objects` must be stored in `θ` after using the `inner` keyword
argument of `repeat()` (see example below).

# Examples
```
K = 6
N = 3
τ = rand(K)
ρ = rand(N)
ν = rand(N)
S = expandgrid(1:9, 1:9)
D = pairwise(Euclidean(), S, S, dims = 1)
L = maternchols(D, ρ, ν)
ρ = repeat(ρ, inner = K ÷ N)
ν = repeat(ν, inner = K ÷ N)
θ = hcat(τ, ρ, ν)'
objectindices(L, θ)
```
"""
function objectindices(objects, θ::AbstractMatrix{T}) where T

	K = size(θ, 2)
	N = size(objects)[end]
	@assert K % N == 0 "The number parameters in θ is not a multiple of the number of objects"

	return repeat(1:N, inner = K ÷ N)
end

end
