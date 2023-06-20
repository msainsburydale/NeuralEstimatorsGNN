module NeuralEstimatorsGNN

using NeuralEstimators
using LinearAlgebra: diagind
using Flux
using Flux: @functor, glorot_uniform
using GraphNeuralNetworks
using GraphNeuralNetworks: check_num_nodes
using Random: seed!
using Statistics: mean
using Distributions #for random simulations
export seed!

export WeightedGraphConv
export DeepSetPool
export adjacencymatrix
export reshapedataDNN, reshapedataGNN, reshapedataCNN
export variableirregularsetup, irregularsetup
export Parameters
export addsingleton
export coverage

"""
	coverage(intervals::V, θ) where  {V <: AbstractArray{M}} where M <: AbstractMatrix

Given a p×K matrix of true parameters `θ`, determine the empirical coverage of
a collection of confidence `intervals` (a K-vector of px2 matrices).

The overall empirical coverage is obtained by averaging the resulting 0-1 matrix
elementwise over all parameter vectors.

# Examples
```
using NeuralEstimators
p = 3
K = 100
θ = rand(p, K)
intervals = [rand(p, 2) for _ in 1:K]
coverage(intervals, θ)
```
"""
function coverage(intervals::V, θ) where  {V <: AbstractArray{M}} where M <: AbstractMatrix

    p, K = size(θ)
	@assert length(intervals) == K
	@assert all(size.(intervals, 1) .== p)
	@assert all(size.(intervals, 2) .== 2)

	# for each confidence interval, determine if the true parameters, θ, are
	# within the interval.
	within = map(eachindex(intervals)) do k

		c = intervals[k]

		# Determine if the confidence intervals contain the true parameter.
		# The result is an indicator vector specifying which parameters are
		# contained in the interval
		[c[i, 1] < θ[i, k] < c[i, 2] for i ∈ 1:p]
	end

	# combine the counts into a single matrix p x K matrix
	within = hcat(within...)

	# compute the empirical coverage
	cvg = mean(within, dims = 2)

	return cvg
end



# ---- WeightedGraphConv ----

@doc raw"""
    WeightedGraphConv(in => out, σ=identity; aggr=+, bias=true, init=glorot_uniform)
Same as regular `GraphConv` layer, but weights the neighbours by spatial distance.

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

#TODO 3D array version of this
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
# universal pooling that I previously encountered also applied DeepSets
# architecture over each graph (that is how I originally thought that it would
# be implemented).
@doc raw"""
    DeepSetPool(ψ, ϕ)
Deep Set readout layer from the paper ['Universal Readout for Graph Convolutional Neural Networks'](https://ieeexplore.ieee.org/document/8852103).
It takes the form,
```math
``\mathbf{h}_V`` = ϕ(|V|⁻¹ \sum_{v\in V} ψ(\mathbf{h}_v)),
```
where ``\mathbf{h}_V`` denotes the summary vector for graph ``V``,
``\mathbf{h}_v`` denotes the vector of hidden features for node ``v \in V``,
and `ψ` and `ψ` are neural networks.

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

# Input graph containing subgraphs (replicating the output of a propagation module)
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

#NB could easily parallelise this to speed it up

"""
	adjacencymatrix(M::Matrix, k::Integer)
	adjacencymatrix(M::Matrix, d::Float)

Computes a spatially weighted adjacency matrix from `M` based on either the `k`
nearest neighbours to each location, or a spatial radius of `d` units.

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
d = 0.3

# Memory efficient constructors (avoids constructing the full distance matrix D)
adjacencymatrix(S, k)
adjacencymatrix(S, d)

# Construct from full distance matrix D
D = pairwise(Euclidean(), S, S, dims = 1)
adjacencymatrix(D, k)
adjacencymatrix(D, d)
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

function adjacencymatrix(M::Mat, d::F) where Mat <: AbstractMatrix{T} where {T, F <: AbstractFloat}

	@assert d > 0

	n = size(M, 1)
	m = size(M, 2)

	if m == n

		D = M
		# bit-matrix specifying which locations are d-neighbours
		A = D .< d
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

			# Find the d-neighbours of s
			j = d .< d
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


# @testset "adjacencymatrix" begin
# 	n = 10
# 	S = rand(n, 2)
# 	k = 5
# 	d = 0.3
# 	A₁ = adjacencymatrix(S, k)
# 	@test all([A₁[i, i] for i ∈ 1:n] .== zeros(n))
# 	A₂ = adjacencymatrix(S, d)
# 	@test all([A₂[i, i] for i ∈ 1:n] .== zeros(n))
#
# 	D = pairwise(Euclidean(), S, S, dims = 1)
# 	Ã₁ = adjacencymatrix(D, k)
# 	Ã₂ = adjacencymatrix(D, d)
# 	@test Ã₁ == A₁
# 	@test Ã₂ == A₂
# end


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


# ---- Parameters definitions and constructors ----

# This is concretely typed so that simulate(params::Parameters, ξ, m::R) is
# type stable. Note that chol_pointer[i] gives the Cholesky factor associated
# with parameter configuration θ[:, i].
struct Parameters{T, I} <: ParameterConfigurations
	θ::Matrix{T}
	chols
	chol_pointer::Vector{I}
end

function Parameters(K::Integer, ξ; J::Integer = 1)

	# All parameters not associated with the Gaussian process
	θ = [rand(ϑ, K * J) for ϑ in drop(ξ.Ω, (:ρ, :ν, :σ))]

	# Determine if we are estimating ν and σ
	estimate_ν = "ν" ∈ ξ.parameter_names
	estimate_σ = "σ" ∈ ξ.parameter_names

	# Covariance parameters associated with the Gaussian process
	ρ = rand(ξ.Ω.ρ, K)
	ν = estimate_ν ? rand(ξ.Ω.ν, K) : fill(ξ.ν, K)
	σ = estimate_σ ? rand(ξ.Ω.σ, K) : fill(ξ.σ, K)
	chols = maternchols(ξ.D, ρ, ν, σ.^2; stack = false)
	chol_pointer = repeat(1:K, inner = J)
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

	Parameters(θ, chols, chol_pointer)
end


# ---- Reshaping data to the correct form ----

function reshapedataCNN(Z)
	n = size(Z[1], 1)
	@assert sqrt(n) == isqrt(n) # assumes a square domain
	reshape.(Z, isqrt(n), isqrt(n), 1, :)
end

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


# Here v is a vector of graphs
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

"""
	addsingleton(x; dim)

# Examples
```
x = rand(4, 4, 10)
addsingleton(x; dim = 3)
```
"""
addsingleton(x; dim) = reshape(x, size(x)[1:dim-1]..., 1, size(x)[dim:end]...)

using Flux: flatten

function reshapedataGNN2(z::A, g::GNNGraph) where A <: AbstractArray{T, N} where {T, N}
	# First, flatten the multi-dimensional array into a matrix, where the first
	# dimension stores the node-level data and the second dimensions stores
	# the replicates. Then, since ndata wants final dimension to equal the
	# number of nodes in the graph, permute the dimensions. Finally, add a
	# singleton first dimension (indicating that we have univariate data)
	z = permutedims(flatten(z))
	z = addsingleton(z; dim = 1)
	GNNGraph(g, ndata = z)
end

function reshapedataGNN2(Z::V, g::GNNGraph) where {V <: AbstractVector{A}} where A <: AbstractArray{T, N} where {T, N}
	reshapedataGNN2.(Z, Ref(g))
end

# Here v is a vector of graphs
function reshapedataGNN2(Z, v::V) where {V <: AbstractVector{A}} where A
	@assert length(Z) == length(v)
	l = length(Z)
	map(1:l) do j
		z = Z[j]
		m = size(z)[end]
		colons  = ntuple(_ -> (:), ndims(z) - 1)
		Flux.batch([GNNGraph(v[j], ndata = (Matrix(vec(z[colons..., i])'))) for i ∈ 1:m])
	end
end



# ---- Setting up for training ----

function irregularsetup(ξ, g; K::Integer, m, J::Integer = 5)

	θ = Parameters(K, ξ, J = J)
	Z = [simulate(θ, mᵢ) for mᵢ ∈ m]
	Z = reshapedataGNN.(Z, Ref(g))

	return θ, Z
end


# Note that neighbour_parameter can be a float or an integer
# note that clustering is set to false by default for backwards compatability
function variableirregularsetup(ξ, n::R; K::Integer, m, J::Integer = 5, return_ξ::Bool = false, neighbour_parameter, clustering::Bool = false) where {R <: AbstractRange{I}} where I <: Integer

	λ_prior = Uniform(10, 90) # λ is uniform between 10 and 90

	# Generate spatial configurations
	S = map(1:K) do k
		nₖ = rand(n)
		if clustering
			λ = rand(λ_prior)
			μ = nₖ / λ
			S = maternclusterprocess(λ = λ, μ = μ)
		else
			S = rand(nₖ, 2)
		end
		S
	end

	# Compute distance matrices
	D = pairwise.(Ref(Euclidean()), S, S, dims = 1)
	A = adjacencymatrix.(D, neighbour_parameter)
	g = GNNGraph.(A)

	ξ = (ξ..., S = S, D = D) # update ξ to contain the new distance matrices (note that Parameters can handle a vector of distance matrices because of maternchols())
	θ = Parameters(K, ξ, J = J)
	Z = [simulate(θ, mᵢ) for mᵢ ∈ m]

	# g = repeat(g, inner = J)
	# Z = reshapedataGNN.(Z, Ref(g)) # FIXME error here

	g = repeat(g, inner = J)
	Z = reshapedataGNN.(Z, Ref(g)) # FIXME error here

	return_ξ ? (θ, Z, ξ) : (θ, Z)
end
variableirregularsetup(ξ, n::Integer; K::Integer, m, J::Integer = 5, return_ξ::Bool = false, neighbour_parameter, clustering::Bool = false) = variableirregularsetup(ξ, range(n, n); K = K, m = m, J = J, return_ξ = return_ξ, neighbour_parameter = neighbour_parameter, clustering = clustering)




# ---- Spatial point process ----

#NB One may also use the R package spatstat using RCall.

export maternclusterprocess

"""
	maternclusterprocess(; λ = 10, μ = 10, r = 0.1, xmin=0, xmax=1, ymin=0, ymax=1)

Simulates a Matérn cluster process with density of parent Poisson point process
`λ`, mean number of daughter points `μ`, and radius of cluster disk `r`, over the
simulation window defined by `{x/y}min` and `{x/y}max`.

# Examples
```
using UnicodePlots

S = maternclusterprocess()
scatterplot(S[:, 1], S[:, 2])

n = 250
λ = [10, 25, 50, 90]
μ = n ./ λ
plots = map(eachindex(λ)) do i
	S = maternclusterprocess(λ = λ[i], μ = μ[i])
	scatterplot(S[:, 1], S[:, 2])
end
```
"""
function maternclusterprocess(; λ = 10, μ = 10, r = 0.1, xmin = 0, xmax = 1, ymin = 0, ymax = 1)

	#Extended simulation windows parameters
	rExt=r #extension parameter -- use cluster radius
	xminExt=xmin-rExt
	xmaxExt=xmax+rExt
	yminExt=ymin-rExt
	ymaxExt=ymax+rExt
	#rectangle dimensions
	xDeltaExt=xmaxExt-xminExt
	yDeltaExt=ymaxExt-yminExt
	areaTotalExt=xDeltaExt*yDeltaExt #area of extended rectangle

	#Simulate Poisson point process
	numbPointsParent=rand(Poisson(areaTotalExt*λ)) #Poisson number of points

	#x and y coordinates of Poisson points for the parent
	xxParent=xminExt.+xDeltaExt*rand(numbPointsParent)
	yyParent=yminExt.+yDeltaExt*rand(numbPointsParent)

	#Simulate Poisson point process for the daughters (ie final poiint process)
	numbPointsDaughter=rand(Poisson(μ),numbPointsParent)
	numbPoints=sum(numbPointsDaughter) #total number of points

	#Generate the (relative) locations in polar coordinates by
	#simulating independent variables.
	theta=2*pi*rand(numbPoints) #angular coordinates
	rho=r*sqrt.(rand(numbPoints)) #radial coordinates

	#Convert polar to Cartesian coordinates
	xx0=rho.*cos.(theta)
	yy0=rho.*sin.(theta)

	#replicate parent points (ie centres of disks/clusters)
	xx=vcat(fill.(xxParent, numbPointsDaughter)...)
	yy=vcat(fill.(yyParent, numbPointsDaughter)...)

	#Shift centre of disk to (xx0,yy0)
	xx=xx.+xx0
	yy=yy.+yy0

	#thin points if outside the simulation window
	booleInside=((xx.>=xmin).&(xx.<=xmax).&(yy.>=ymin).&(yy.<=ymax))
	xx=xx[booleInside]
	yy=yy[booleInside]

	hcat(xx, yy)
end


#module
end
