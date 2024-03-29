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

# Method that automatically constructs spatial locations
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

	# Pass these objects into the next constructor
	Parameters(K, (ξ..., S = S); J = J)
end

# Method that requires the spatial locations, S, to be stored in ξ
function Parameters(K::Integer, ξ; J::Integer = 1)

	@assert :Ω ∈ keys(ξ)
	@assert :S ∈ keys(ξ)
	@assert :neighbourhood ∈ keys(ξ)
	@assert :δ ∈ keys(ξ) || :k ∈ keys(ξ)
	@assert :σ ∈ union(keys(ξ), keys(ξ.Ω))
	@assert :ν ∈ union(keys(ξ), keys(ξ.Ω))

	S = ξ.S
	if !(typeof(S) <: AbstractVector) S = [S] end
	D = pairwise.(Ref(Euclidean()), S, dims = 1)

	@assert length(S) ∈ (1, K)
	loc_pointer = length(S) == 1 ? repeat([1], K*J) : repeat(1:K, inner = J)

	if ξ.neighbourhood == "fixedradius"
		A = adjacencymatrix.(S, ξ.δ)
	elseif ξ.neighbourhood == "knearest"
		A = adjacencymatrix.(S, ξ.k)
	elseif ξ.neighbourhood == "combined"
		A = adjacencymatrix.(S, ξ.δ, ξ.k)
	elseif ξ.neighbourhood == "maxmin"
		A = adjacencymatrix.(S, ξ.k, maxmin = true)
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


function Parameters(θ, S, ξ)

	K = size(θ, 2)

	@assert :Ω ∈ keys(ξ)
	@assert :neighbourhood ∈ keys(ξ)
	@assert :δ ∈ keys(ξ) || :k ∈ keys(ξ)
	@assert :σ ∈ union(keys(ξ), keys(ξ.Ω))
	@assert :ν ∈ union(keys(ξ), keys(ξ.Ω))

	if !(typeof(S) <: AbstractVector) S = [S] end
	D = pairwise.(Ref(Euclidean()), S, dims = 1)

	@assert length(S) ∈ (1, K)
	loc_pointer = length(S) == 1 ? repeat([1], K) : collect(1:K)

	if ξ.neighbourhood == "fixedradius"
		A = adjacencymatrix.(S, ξ.δ)
	elseif ξ.neighbourhood == "knearest"
		A = adjacencymatrix.(S, ξ.k)
	elseif ξ.neighbourhood == "combined"
		A = adjacencymatrix.(S, ξ.δ, ξ.k)
	elseif ξ.neighbourhood == "maxmin"
		A = adjacencymatrix.(S, ξ.k, maxmin = true)
	else
		error("ξ.neighbourhood = $(ξ.neighbourhood) not a valid choice for the neighbourhood definition; please use either 'fixedradius', 'knearest', or 'combined'.")
	end
	graphs = GNNGraph.(A)

	# Find indices for ρ (and possibly ν and σ) with respect to θ.
	parameter_names = String.(collect(keys(ξ.Ω)))
	ρ_idx = findfirst(parameter_names .== "ρ")
	ν_idx = findfirst(parameter_names .== "ν")
	σ_idx = findfirst(parameter_names .== "σ")

	# Compute Cholesky factors
	ρ = isnothing(ρ_idx) ? fill(ξ.ρ, K) : θ[ρ_idx, :]
	ν = isnothing(ν_idx) ? fill(ξ.ν, K) : θ[ν_idx, :]
	σ = isnothing(σ_idx) ? fill(ξ.σ, K) : θ[σ_idx, :]
	chols = maternchols(D, ρ, ν, σ.^2; stack = false)
	chol_pointer = collect(1:K)

	# Convert to Float32 for efficiency
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

function modifyneighbourhood(θ::Parameters, k::Integer; kwargs...)

	S = θ.locations
	if !(typeof(S) <: AbstractVector) S = [S] end

	A = adjacencymatrix.(S, k; kwargs...)
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
