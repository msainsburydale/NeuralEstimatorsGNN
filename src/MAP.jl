using LinearAlgebra
using NeuralEstimators
using Optim
using Folds
using Flux: flatten

function MAP(Z::V, ξ) where {T, N, A <: AbstractArray{T, N}, V <: AbstractVector{A}}

	# Compress the data from an n-dimensional array to a matrix
	Z = flatten.(Z)

	# intitialise the estimates to the true parameters. Since we logistic-transform
	# the parameters during optimisation to force the estimates to be within the
	# prior support, here we provide the logit-transformed values.
	Ω = ξ.Ω
	Ω = [Ω...] # convert to array since broadcasting over dictionaries and NamedTuples is reserved
	θ₀ = scaledlogit.(ξ.θ₀, Ω)

	# Convert to Float64 so that Cholesky factorisation doesn't throw positive
	# definite error due to rounding.
	# (When you want to broadcast broadcast, then broadcast broadcast)
	Z  = broadcast.(x -> !ismissing(x) ? Float64(x) : identity(x), Z)
	θ₀ = Float64.(θ₀)

	# If Z is replicated, try to replicate θ₀ accordingly.
	K = size(θ₀, 2)
	m = length(Z)
	if m != K
		if (m ÷ K) == (m / K)
			θ₀ = repeat(θ₀, outer = (1, m ÷ K))
		else
			error("The length of the data vector, m = $m, and the number of parameter configurations, K = $K, do not match; further, m is not a multiple of K, so we cannot replicate θ to match Z.")
		end
		K = size(θ₀, 2) # update K
	end

	# Distance matrix: D may be a single matrix or a vector of matrices
	D = ξ.D
	if typeof(D) <: AbstractVector
		# If θ₀ is replicated, try to create an approporiate pointer for D based
		# on same way that chol_pointer is constructed.
		L = length(D)
		if L != K && (K ÷ L) != (K / L)
			error("The number of parameter configurations, K = $K, and the number of distances matrics, L = $L, do not match; further, K is not a multiple of L, so we cannot replicate D to match θ.")
		end
		D_pointer = repeat(1:L, inner = K ÷ L) # Note that this is exactly the same as the field chol_pointer in objects of type Parameters
	else
		D = [D]
		D_pointer = repeat([1], K)
	end
	@assert length(D_pointer) == K

	# Convert from matrix to vector of vectors
	θ₀ = [θ₀[:, k] for k ∈ 1:K]

	# Optimise
	θ̂ = Folds.map(1:K) do k
		 Dₖ = D[D_pointer[k]]
		 MAP(Z[k], θ₀[k], Dₖ, Ω)
	end

	# Convert to matrix
	θ̂ = hcat(θ̂...)

	return θ̂
end



function MAP(Z::M, θ₀::V, D, Ω) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}

	# Closure that will be minimised
	loss(θ) = nll(θ, Z, D, Ω)

	# Estimate the parameters
	θ̂ = optimize(loss, θ₀, NelderMead()) |> Optim.minimizer

	# During optimisation, we constrained the parameters using the scaled-logistic
	# function; here, we convert to the orginal scale
	θ̂ = scaledlogistic.(θ̂, Ω)

	return θ̂
end
