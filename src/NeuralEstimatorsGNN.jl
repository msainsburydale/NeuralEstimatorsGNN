module NeuralEstimatorsGNN

using NeuralEstimators
using Flux
using Flux: flatten
using GraphNeuralNetworks
using Random: seed!; export seed!
using Distributions
using Distances

export addsingleton
export spatialconfigurations
export Parameters, modifyneighbourhood
export reshapedataGNN, reshapedataGNNcompact



# ---- adjacencymatrix ----

import NeuralEstimators: adjacencymatrix

using InvertedIndices
using NearestNeighbors
using StatsBase: sample
using BenchmarkTools
using SparseArrays

function adjacencymatrix(M::Mat, k::Integer; maxmin::Bool = false, moralise::Bool = false) where Mat <: AbstractMatrix{T} where T

	@assert k > 0

	I = Int64[]
	J = Int64[]
	V = Float64[]
	n = size(M, 1)
	m = size(M, 2)

	if m == n # square matrix, so assume M is a distance matrix
		D = M
	else      # otherwise, M is a matrix of spatial locations
		S = M
	end

	if k >= n # more neighbours than observations: return a dense adjacency matrix
		if m != n
			D = pairwise(Euclidean(), S')
		end
		A = sparse(D)
	elseif !maxmin
		k += 1 # each location neighbours itself, so increase k by 1
		for i ∈ 1:n

			if m == n
				d = D[i, :]
			else
				# Compute distances between sᵢ and all other locations
				d = colwise(Euclidean(), S', S[i, :])
			end

			# Find the neighbours of s
			j, v = findneighbours(d, k)

			push!(I, repeat([i], inner = k)...)
			push!(J, j...)
			push!(V, v...)
		end
		A = sparse(I,J,V,n,n)
	else
		@assert m != n "`adjacencymatrix` with maxmin-ordering requires a matrix of spatial locations, not a distance matrix"
		ord     = ordermaxmin(S)          # calculate ordering
		Sord    = S[ord, :]               # re-order locations
		NNarray = findorderednn(Sord, k)  # find k nearest neighbours/"parents"
		R = builddag(NNarray)             # build DAG
		A = moralise ?  R' * R : R        # moralise

		# Add distances to A
		# NB This is inefficient, especially for large n; only optimise
		#    if we find that this approach works well
		D = pairwise(Euclidean(), Sord')
		I, J, V = findnz(A)
		indices = collect(zip(I,J))
		indices = CartesianIndex.(indices)
		A.nzval .= D[indices]

		# "unorder" back to the original ordering
		# Sanity check: Sord[sortperm(ord), :] == S
		# Sanity check: D[sortperm(ord), sortperm(ord)] == pairwise(Euclidean(), S')
		A = A[sortperm(ord), sortperm(ord)]
	end

	return A
end

function findneighbours(d, k::Integer)
	V = partialsort(d, 1:k)
	J = [findfirst(v .== d) for v ∈ V]
    return J, V
end

function getknn(S, s, k; args...)
  tree = KDTree(S; args...)
  nn_index, nn_dist = knn(tree, s, k, true)
  nn_index = hcat(nn_index...) |> permutedims # nn_index = stackarrays(nn_index, merge = false)'
  nn_dist  = hcat(nn_dist...)  |> permutedims # nn_dist  = stackarrays(nn_dist, merge = false)'
  nn_index, nn_dist
end

function ordermaxmin(S)

  # get number of locs
  n = size(S, 1)
  k = isqrt(n)
  # k is number of neighbors to search over
  # get the past and future nearest neighbors
  NNall = getknn(S', S', k)[1]
  # pick a random ordering
  index_in_position = [sample(1:n, n, replace = false)..., repeat([missing],1*n)...]
  position_of_index = sortperm(index_in_position[1:n])
  # loop over the first n/4 locations
  # move an index to the end if it is a
  # near neighbor of a previous location
  curlen = n
  nmoved = 0
  for j ∈ 2:2n
	nneigh = round(min(k, n /(j-nmoved+1)))
    nneigh = Int(nneigh)
    neighbors = NNall[index_in_position[j], 1:nneigh]
    if minimum(skipmissing(position_of_index[neighbors])) < j
      nmoved += 1
      curlen += 1
      position_of_index[ index_in_position[j] ] = curlen
      rassign(index_in_position, curlen, index_in_position[j])
      index_in_position[j] = missing
  	end
  end
  ord = collect(skipmissing(index_in_position))

  return ord
end

function rassign(v::AbstractVector, index::Integer, x)
	@assert index > 0
	if index <= length(v)
		v[index] = x
	elseif index == length(v)+1
		push!(v, x)
	else
		v = [v..., fill(missing, index - length(v) - 1)..., x]
	end
	return v
end

function findorderednnbrute(S, k::Integer)
  # find the k+1 nearest neighbors to S[j,] in S[1:j,]
  # by convention, this includes S[j,], which is distance 0
  n = size(S, 1)
  k = min(k,n-1)
  NNarray = Matrix{Union{Integer, Missing}}(missing, n, k+1)
  for j ∈ 1:n
	d = colwise(Euclidean(), S[1:j, :]', S[j, :])
    NNarray[j, 1:min(k+1,j)] = sortperm(d)[1:min(k+1,j)]
  end
  return NNarray
end

function findorderednn(S, k::Integer)

  # number of locations
  n = size(S, 1)
  k = min(k,n-1)
  mult = 2

  # to store the nearest neighbor indices
  NNarray = Matrix{Union{Integer, Missing}}(missing, n, k+1)

  # find neighbours of first mult*k+1 locations by brute force
  maxval = min( mult*k + 1, n )
  NNarray[1:maxval, :] = findorderednnbrute(S[1:maxval, :],k)

  query_inds = min( maxval+1, n):n
  data_inds = 1:n
  ksearch = k
  while length(query_inds) > 0
    ksearch = min(maximum(query_inds), 2ksearch)
    data_inds = 1:min(maximum(query_inds), n)
	NN = getknn(S[data_inds, :]', S[query_inds, :]', ksearch)[1]

    less_than_l = hcat([NN[l, :] .<= query_inds[l] for l ∈ 1:size(NN, 1)]...) |> permutedims
	sum_less_than_l = vec(mapslices(sum, less_than_l, dims = 2))
    ind_less_than_l = findall(sum_less_than_l .>= k+1)
	NN_k = hcat([NN[l,:][less_than_l[l,:]][1:(k+1)] for l ∈ ind_less_than_l]...) |> permutedims
    NNarray[query_inds[ind_less_than_l], :] = NN_k

    query_inds = query_inds[Not(ind_less_than_l)]
  end

  return NNarray
end

function builddag(NNarray)
  n, k = size(NNarray)
  I = [1]
  J = [1]
  V = Float64[1.0]
  for j in 2:n
    i = NNarray[j, :]
    i = collect(skipmissing(i))
    push!(J, repeat([j], length(i))...)
    push!(I, i...)
	push!(V, repeat([1], length(i))...)
  end
  R = sparse(I,J,V,n,n)
  return R
end

# n=100
# S = rand(n, 2)
# k=5
# ord = ordermaxmin(S)              # calculate maxmin ordering
# Sord = S[ord, :];                 # reorder locations
# NNarray = findorderednn(Sord, k)  # find k nearest neighbours/"parents"
# R = builddag(NNarray)             # build the DAG
# Q = R' * R                        # moralise



# ---- Utility functions ----

"""
	addsingleton(x; dim)

Adds a singleton dimension to `x` at dimension `dim`.

# Examples
```
x = rand(4, 4, 10)
addsingleton(x; dim = 3)
```
"""
addsingleton(x; dim) = reshape(x, size(x)[1:dim-1]..., 1, size(x)[dim:end]...)


"""
	spatialconfigurations(n::Integer, set::String)
Generates spatial configurations of size `n` corresponding to one of
the four types of `set`s as used in Section 3 of the manuscript.

# Examples
```
n = 250
S₁ = spatialconfigurations(n, "uniform")
S₂ = spatialconfigurations(n, "quadrants")
S₃ = spatialconfigurations(n, "mixedsparsity")
S₄ = spatialconfigurations(n, "cup")

using UnicodePlots
[scatterplot(S[:, 1], S[:, 2]) for S ∈ [S₁, S₂, S₃, S₄]]
```
"""
function spatialconfigurations(n::Integer, set::String)

	@assert n > 0
	@assert set ∈ ["uniform", "quadrants", "mixedsparsity", "cup"]

	if set == "uniform"
		S = rand(n, 2)
	elseif set == "quadrants"
		S₁ = 0.5 * rand(n÷2, 2)
		S₂ = 0.5 * rand(n÷2, 2) .+ 0.5
		S  = vcat(S₁, S₂)
	elseif set == "mixedsparsity"
		n_centre = (3 * n) ÷ 4
		n_corner = (n - n_centre) ÷ 4
		S_centre  = 1/3 * rand(n_centre, 2) .+ 1/3
		S_corner1 = 1/3 * rand(n_corner, 2)
		S_corner2 = 1/3 * rand(n_corner, 2); S_corner2[:, 2] .+= 2/3
		S_corner3 = 1/3 * rand(n_corner, 2); S_corner3[:, 1] .+= 2/3
		S_corner4 = 1/3 * rand(n_corner, 2); S_corner4 .+= 2/3
		S = vcat(S_centre, S_corner1, S_corner2, S_corner3, S_corner4)
	elseif set == "cup"
		n_strip2 = n÷3 + n % 3 # ensure that total sample size is n (even if n is not divisible by 3)
		S_strip1 = rand(n÷3, 2);      S_strip1[:, 1] .*= 0.2;
		S_strip2 = rand(n_strip2, 2); S_strip2[:, 1] .*= 0.6; S_strip2[:, 1] .+= 0.2; S_strip2[:, 2] .*= 1/3;
		S_strip3 = rand(n÷3, 2);      S_strip3[:, 1] .*= 0.2; S_strip3[:, 1] .+= 0.8;
		S = vcat(S_strip1, S_strip2, S_strip3)
	end

	return S
end

# ---- Parameters definitions and constructors ----

"""
Type for storing parameter configurations, with fields storing the matrix of
parameters (`θ`), the Cholesky factors associated with these parameters (`chols`),
and a pointer (`chol_pointer`) where `chol_pointer[i]` gives the Cholesky factor
associated with parameter configuration `θ[:, i]`.

The convencience constructor `Parameters(K::Integer, ξ; J::Integer = 1)` is the
intended way for the objects to be constructed. Here, `K` is the number of unique
Gaussian-Process (GP) parameters to be sampled (see below) and, hence, the number of
Cholesky factors that need to be computed; `KJ` is the total number of parameters
to be sampled (the GP parameters will be repeated `J` times), and `ξ` is a
named tuple containing fields:

- `Ω`: the prior distribution, itself a named tuple where each field can be sampled using rand(),
- `S`: matrix of spatial coordinates (or K-vector of matrices),
- `D`: distance matrix (or K-vector of matrices),
- `δ`: fixed radius when constructing the neighbour matrix,
- `k`: maximum number of neighbours to consider when constructing the neighbour matrix.

The type assumes the presence of a GP in the model, with range parameter
ρ, smoothness ν, and marginal standard deviation σ; if the latter two parameters
are not being estimated (i.e., they are fixed and known), then they should not be
in the prior Ω and single values for each should instead be stored in
`ξ.ν` and `ξ.σ`, respectively.
"""
struct Parameters{T, I} <: ParameterConfigurations
	θ::Matrix{T}
	locations
	graphs
	chols
	chol_pointer::Vector{I}
	loc_pointer::Vector{I}
end

# Method that automatically constructs spatial locations from a Matern cluster
# process, or uniformly sampled.
function Parameters(K::Integer, ξ, n; J::Integer = 1, cluster_process::Bool = true)

	if typeof(n) <: Integer
		n = range(n, n)
	end

	# Simulate spatial locations from a cluster process over the unit square
	if cluster_process
		λ_prior = Uniform(10, 100) # λ_prior = Uniform(ceil(n/20), ceil(n/3))
		S = map(1:K) do k
			nₖ = rand(n)
			λₖ = rand(λ_prior)
			μₖ = nₖ / λₖ
			Sₖ = maternclusterprocess(λ = λₖ, μ = μₖ)
			Sₖ
		end
	else
		S = [rand(rand(n), 2) for k ∈ 1:K]
	end

	# Compute distance matrices and construct the graphs
	D = pairwise.(Ref(Euclidean()), S, S, dims = 1)

	# Pass these objects into the next constructor
	Parameters(K, (ξ..., S = S, D = D); J = J)
end

# Method that assumes the spatial locations, S, and distance matrices, D, are
# stored in ξ
function Parameters(K::Integer, ξ; J::Integer = 1)

	@assert :Ω ∈ keys(ξ)
	@assert :D ∈ keys(ξ)
	@assert :S ∈ keys(ξ)
	@assert :neighbourhood ∈ keys(ξ)
	@assert :δ ∈ keys(ξ) || :k ∈ keys(ξ)
	@assert :σ ∈ union(keys(ξ), keys(ξ.Ω))
	@assert :ν ∈ union(keys(ξ), keys(ξ.Ω))

	D = ξ.D
	S = ξ.S

	if !(typeof(D) <: AbstractVector) D = [D] end
	if !(typeof(S) <: AbstractVector) S = [S] end

	@assert length(S) ∈ (1, K)
	@assert length(D) ∈ (1, K)
	@assert length(S) == length(D)
	loc_pointer = length(S) == 1 ? repeat([1], K*J) : repeat(1:K, inner = J)

	if ξ.neighbourhood == "fixedradius"
		A = adjacencymatrix.(D, ξ.δ)
	elseif ξ.neighbourhood == "knearest"
		A = adjacencymatrix.(D, ξ.k)
	elseif ξ.neighbourhood == "combined"
		A = adjacencymatrix.(D, ξ.δ, ξ.k)
	else
		error("ξ.neighbourhood = $(ξ.neighbourhood) not a valid choice for the neighbourhood definition; please use either 'fixedradius', 'knearest', or 'combined'.")
	end
	graphs = GNNGraph.(A)

	# Sample parameters not associated with the Gaussian process
	θ = [rand(ϑ, K * J) for ϑ in drop(ξ.Ω, (:ρ, :ν, :σ))]

	# Sample parameters from the Gaussian process and compute Cholesky factors
	ρ = rand(ξ.Ω.ρ, K)
	parameter_names = String.(collect(keys(ξ.Ω)))
	estimate_ν = "ν" ∈ parameter_names
	estimate_σ = "σ" ∈ parameter_names
	ν = estimate_ν ? rand(ξ.Ω.ν, K) : fill(ξ.ν, K)
	σ = estimate_σ ? rand(ξ.Ω.σ, K) : fill(ξ.σ, K)
	chols = maternchols(D, ρ, ν, σ.^2; stack = false)
	chol_pointer = repeat(1:K, inner = J)
	ρ = repeat(ρ, inner = J)
	ν = repeat(ν, inner = J)
	σ = repeat(σ, inner = J)

	# Insert ρ (and possibly ν and σ) into θ.
	ρ_idx = findfirst(parameter_names .== "ρ")
	ν_idx = findfirst(parameter_names .== "ν")
	σ_idx = findfirst(parameter_names .== "σ")
	θ₁ = θ[1:(ρ_idx-1)]
	θ₂ = θ[ρ_idx:end] # Note that ρ and ν are not in θ, so we don't need to skip any indices.
	#NB Would be better to just place the parameters in the right place using their _idx value, rather than imposing an artificial ordering.
	if estimate_ν && estimate_σ
		@assert (ν_idx == ρ_idx + 1) && (σ_idx == ν_idx + 1) "The code assumes that ρ, ν, and σ are stored continguously in the prior Ω, and in that order"
		θ = [θ₁..., ρ, ν, σ, θ₂...]
	elseif estimate_ν
		@assert ν_idx == ρ_idx + 1 "The code assumes that ρ and ν are stored continguously in the prior Ω, and in that order"
		θ = [θ₁..., ρ, ν, θ₂...]
	elseif estimate_σ
		@assert σ_idx == ρ_idx + 1 "The code assumes that ρ and σ are stored continguously in the prior Ω, and in that order"
		θ = [θ₁..., ρ, σ, θ₂...]
	else
		θ = [θ₁..., ρ, θ₂...]
	end

	# Combine parameters into a pxK matrix
	θ = permutedims(hcat(θ...))
	θ = Float32.(θ)

	Parameters(θ, S, graphs, chols, chol_pointer, loc_pointer)
end

# Same API as adjacencymatrix()
function modifyneighbourhood(θ::Parameters, args...)

	S = θ.locations
	if !(typeof(S) <: AbstractVector) S = [S] end

	A = adjacencymatrix.(S, args...)
	graphs = GNNGraph.(A)

	Parameters(θ.θ, S, graphs, θ.chols, θ.chol_pointer, θ.loc_pointer)
end


# ---- Reshaping data to the correct form for a GNN ----

"""
	reshapedataGNN(Z, g::GNNGraph)
	reshapedataGNN(Z, g::V) where {V <: AbstractVector{G}} where G <: GNNGraph

Merges the vector of data sets `Z` with the graph(s) `g`.

Each data set in `Z` should be stored as an array with final dimension storing
the replicates dimension; this implies that each data set is replicated and
observed over the same set of spatial locations.
"""
function reshapedataGNN(Z, g::GNNGraph)
	map(Z) do z
		m = size(z)[end]
		colons  = ntuple(_ -> (:), ndims(z) - 1)
		v = [GNNGraph(g, ndata = (Matrix(vec(z[colons..., i])'))) for i ∈ 1:m]
		Flux.batch(v)
	end
end

function reshapedataGNN(Z, v::V) where {V <: AbstractVector{G}} where G # G should be a graph
	@assert length(Z) == length(v)
	map(eachindex(Z)) do j
		z = Z[j]
		m = size(z)[end]
		colons  = ntuple(_ -> (:), ndims(z) - 1)
		Flux.batch([GNNGraph(v[j], ndata = (Matrix(vec(z[colons..., i])'))) for i ∈ 1:m])
	end
end

"""
Same as reshapedataGNN, but stores the data in a more efficient way. This format,
however, is only applicable when the data are replicated and the same set of
spatial locations, and custom implementations of propagation layers from
GraphNeuralNetworks.jl are required.
"""
function reshapedataGNNcompact(z::A, g::GNNGraph) where A <: AbstractArray{T, N} where {T, N}
	# First, flatten the multi-dimensional array into a matrix, where the first
	# dimension stores the node-level data and the second dimensions stores
	# the replicates. Then, since ndata wants final dimension to equal the
	# number of nodes in the graph, permute the dimensions. Finally, add a
	# singleton first dimension (indicating that we have univariate data)
	z = permutedims(flatten(z))
	z = addsingleton(z; dim = 1)
	GNNGraph(g, ndata = z)
end

function reshapedataGNNcompact(Z::V, g::GNNGraph) where {V <: AbstractVector{A}} where A <: AbstractArray{T, N} where {T, N}
	reshapedataGNNcompact.(Z, Ref(g))
end

# Here v is a vector of graphs
function reshapedataGNNcompact(Z, v::V) where {V <: AbstractVector{A}} where A
	@assert length(Z) == length(v)
	l = length(Z)
	map(1:l) do j
		z = Z[j]
		m = size(z)[end]
		colons  = ntuple(_ -> (:), ndims(z) - 1)
		Flux.batch([GNNGraph(v[j], ndata = (Matrix(vec(z[colons..., i])'))) for i ∈ 1:m])
	end
end


end #module
