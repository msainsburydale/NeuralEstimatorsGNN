module NeuralEstimatorsGNN

using NeuralEstimators
using Flux
using Flux: flatten
using GraphNeuralNetworks
using Random: seed!
using Distributions
using Distances

export addsingleton
export variableirregularsetup, irregularsetup
export Parameters
export reshapedataGNN
export seed!

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
	elseif estimate_σ
		θ = [θ₁..., ρ, σ, θ₂...]
	else
		θ = [θ₁..., ρ, θ₂...]
	end

	# Concatenate into a matrix and convert to Float32 for efficiency
	θ = hcat(θ...)'
	θ = Float32.(θ)

	Parameters(θ, chols, chol_pointer)
end


# ---- Reshaping data to the correct form ----

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
# Note that clustering is set to false by default for backwards compatability
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

	# Compute distance matrices and construct the graphs
	D = pairwise.(Ref(Euclidean()), S, S, dims = 1)
	A = adjacencymatrix.(D, neighbour_parameter)
	g = GNNGraph.(A)

	# Update ξ to contain the new distance matrices. Note that Parameters() can
	# handle a vector of distance matrices because maternchols() is able to do it.
	ξ = (ξ..., S = S, D = D)
	θ = Parameters(K, ξ, J = J)
	Z = [simulate(θ, mᵢ) for mᵢ ∈ m]

	g = repeat(g, inner = J)
	Z = reshapedataGNN.(Z, Ref(g))

	return_ξ ? (θ, Z, ξ) : (θ, Z)
end
variableirregularsetup(ξ, n::Integer; K::Integer, m, J::Integer = 5, return_ξ::Bool = false, neighbour_parameter, clustering::Bool = false) = variableirregularsetup(ξ, range(n, n); K = K, m = m, J = J, return_ξ = return_ξ, neighbour_parameter = neighbour_parameter, clustering = clustering)


end #module
