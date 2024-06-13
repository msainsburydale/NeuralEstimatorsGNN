module NeuralEstimatorsGNN

using NeuralEstimators
using Flux
using Folds
using GraphNeuralNetworks
using LinearAlgebra
using Random: seed!; export seed!
using Distributions
using Distances

export Parameters

#TODO probably neater to just forego xi from this functionality...

"""
	Parameters <: ParameterConfigurations
Type for storing parameter configurations, with fields storing the matrix of
parameters (`θ`), the Cholesky factors associated with these parameters (`chols`),
and a pointer (`chol_pointer`) where `chol_pointer[i]` gives the Cholesky factor
associated with parameter configuration `θ[:, i]`.

The convencience constructor `Parameters(K::Integer, ξ; J::Integer = 1)` is the
intended way for the objects to be constructed. Here, `K` is the number of unique
Gaussian-process (GP) parameters to be sampled (see below) and, hence, the number of
Cholesky factors that need to be computed; `KJ` is the total number of parameters
to be sampled (the GP parameters will be repeated `J` times), and `ξ` is a
named tuple containing fields:

- `Ω`: the prior distribution, itself a named tuple where each field can be sampled using rand(),
- `S`: matrix of spatial coordinates (or K-vector of matrices),

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

# Method that simulates spatial configurations
function Parameters(K::Integer, ξ, n; J::Integer = 1, cluster_process::Bool = true)

	if typeof(n) <: Integer
		n = range(n, n)
	end

	# Simulate spatial locations from a cluster process over the unit square
	if cluster_process
		λ_prior = Uniform(10, 100) 
		S = map(1:K) do k
			nₖ = rand(n)
			λₖ = rand(λ_prior)
			μₖ = nₖ / λₖ
			Sₖ = maternclusterprocess(λ = λₖ, μ = μₖ; unit_bounding_box = true)
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
	@assert :σ ∈ union(keys(ξ), keys(ξ.Ω))
	@assert :ν ∈ union(keys(ξ), keys(ξ.Ω))

	S = ξ.S
	if !(typeof(S) <: AbstractVector) S = [S] end
	D = pairwise.(Ref(Euclidean()), S, dims = 1)
	S = broadcast.(Float32, S)

	@assert length(S) ∈ (1, K)
	loc_pointer = length(S) == 1 ? repeat([1], K*J) : repeat(1:K, inner = J)

  r = 0.15 # cutoff distance used to define the neighbourhood of each node
  k = 30   # maximum number of neighbours to consider
	A = adjacencymatrix.(S, r, k)
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
	σ² = σ.^2
	#chols = maternchols(D, ρ, ν, σ²; stack = false)
	D = UpperTriangular.(D) # D is symmetric, so only consider upper triangle
	chols = Folds.map(1:K) do k
	  Dₖ = length(D) == 1 ? D[1] : D[k]
		Σ = matern.(Dₖ, ρ[k], ν[k], σ²[k]) 
		L = cholesky(Symmetric(Σ)).L
		L = convert(Array, L) # convert from Triangular to Array
		L
	end
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
	θ = permutedims(reduce(hcat, θ))

	# Convert to Float32 for efficiency
	θ = Float32.(θ)

	Parameters(θ, S, graphs, chols, chol_pointer, loc_pointer)
end

# Can be used for parametric bootstrap
# TODO A lot of code repetition with the method above.. Can't I just make the above method call this one?
function Parameters(θ, S, ξ)

	K = size(θ, 2)

	@assert :Ω ∈ keys(ξ)
	@assert :σ ∈ union(keys(ξ), keys(ξ.Ω))
	@assert :ν ∈ union(keys(ξ), keys(ξ.Ω))

	if !(typeof(S) <: AbstractVector) S = [S] end
	D = pairwise.(Ref(Euclidean()), S, dims = 1)
	S = broadcast.(Float32, S)

	@assert length(S) ∈ (1, K)
	loc_pointer = length(S) == 1 ? repeat([1], K) : collect(1:K)

  r = 0.15 # disc radius used to define the neighbourhood of each node
	k = 30   # maximum number of neighbours to consider when constructing the neighbourhood
	A = adjacencymatrix.(S, r, k)
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

end #module
