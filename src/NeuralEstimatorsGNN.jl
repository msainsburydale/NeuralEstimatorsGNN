module NeuralEstimatorsGNN

using NeuralEstimators
using Flux
using Flux: flatten
using GraphNeuralNetworks
using Random: seed!
using Distributions
using Distances

export spatialconfigurations , addsingleton
export variableirregularsetup
export Parameters
export reshapedataGNN
export seed!


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
	Parameters(θ::Matrix, chols, chol_pointer::Vector{Integer})

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
- `D`: a distance matrix (or possible a vector of distance matrices of length `K`),

The type assumes the presence of a GP in the model, with range parameter
ρ, smoothness ν, and marginal standard deviation σ; if the latter two parameters
are not being estimated (i.e., they are fixed and known), then they should not be
in the prior Ω and single values for each should instead be stored in
`ξ.ν` and `ξ.σ`, respectively.
"""
struct Parameters{T, I} <: ParameterConfigurations
	θ::Matrix{T}
	chols
	chol_pointer::Vector{I}
end
# This is concretely typed for type stability of simulate().


function Parameters(K::Integer, ξ; J::Integer = 1)

	# All parameters not associated with the Gaussian process
	θ = [rand(ϑ, K * J) for ϑ in drop(ξ.Ω, (:ρ, :ν, :σ))]

	# Determine if we are estimating ν and σ
	# Note that we extract the parameter names here, even though ξ will
	# typically contain the parameter_names too.
	parameter_names = String.(collect(keys(ξ.Ω)))
	estimate_ν = "ν" ∈ parameter_names
	estimate_σ = "σ" ∈ parameter_names

	# GP covariance parameters and Cholesky factors
	ρ = rand(ξ.Ω.ρ, K)
	ν = estimate_ν ? rand(ξ.Ω.ν, K) : fill(ξ.ν, K)
	σ = estimate_σ ? rand(ξ.Ω.σ, K) : fill(ξ.σ, K)
	chols = maternchols(ξ.D, ρ, ν, σ.^2; stack = false)
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

	# Concatenate θ into a matrix and convert to Float32 for efficiency
	θ = hcat(θ...)'
	θ = Float32.(θ)

	Parameters(θ, chols, chol_pointer)
end

function Parameters(θ::Matrix, ξ)

	p, K = size(θ)

	# Determine if we are estimating ν and σ
	parameter_names = String.(collect(keys(ξ.Ω)))
	estimate_ν = "ν" ∈ parameter_names
	estimate_σ = "σ" ∈ parameter_names

	# GP covariance parameters and Cholesky factors
	ρ_idx = findfirst(parameter_names .== "ρ")
	ν_idx = findfirst(parameter_names .== "ν")
	σ_idx = findfirst(parameter_names .== "σ")
	ρ = θ[ρ_idx, :]
	ν = estimate_ν ? θ[ν_idx, :] : fill(ξ.ν, K)
	σ = estimate_σ ? θ[σ_idx, :] : fill(ξ.σ, K)
	chols = maternchols(ξ.D, ρ, ν, σ.^2; stack = false)
	chol_pointer = collect(1:K)

	Parameters(θ, chols, chol_pointer)
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

# ---- Setting up data for training ----


function variableirregularsetup(ξ, n::R; K::Integer, m, J::Integer = 5) where {R <: AbstractRange{I}} where I <: Integer

	λ_prior = Uniform(10, 90) # λ is uniform between 10 and 90

	# Generate spatial configurations
	S = map(1:K) do k
		nₖ = rand(n)
		λ = rand(λ_prior)
		μ = nₖ / λ
		S = maternclusterprocess(λ = λ, μ = μ)
		S
	end

	# Compute distance matrices and construct the graphs
	D = pairwise.(Ref(Euclidean()), S, S, dims = 1)
	A = adjacencymatrix.(D, ξ.δ) # Note that ξ.δ can be a float or an integer, which will give different behaviours
	g = GNNGraph.(A)

	# Update ξ to contain the new distance matrices. Note that Parameters() can
	# handle a vector of distance matrices because maternchols() is able to do it.
	ξ = (ξ..., S = S, D = D)
	parameters = Parameters(K, ξ, J = J)
	Z = [simulate(parameters, mᵢ) for mᵢ ∈ m]

	g = repeat(g, inner = J)
	Z = reshapedataGNN.(Z, Ref(g))

	parameters, Z
end
variableirregularsetup(ξ, n::Integer; K::Integer, m, J::Integer = 5) = variableirregularsetup(ξ, range(n, n); K = K, m = m, J = J)

#TODO remove variableirregularsetup(); we can get the same functionality by storing the graphs in Parameters.


end #module
