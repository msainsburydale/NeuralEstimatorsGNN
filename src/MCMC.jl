using LinearAlgebra
using Folds
using Flux: flatten
using NeuralEstimators

function MCMC(Z::V, ξ) where {T, N, A <: AbstractArray{T, N}, V <: AbstractVector{A}}

	# Initial values
	θ₀ = ξ.θ₀

	# Compress the data from an n-dimensional array to a matrix
	Z = flatten.(Z)

	# inverse of the variance-stabilising transform
	Z = broadcast.(ξ.invtransform, Z)

	# prior
	Ω = ξ.Ω

	# Convert to Float64 so that Cholesky factorisation doesn't throw positive
	# definite error due to rounding.
	Z  = broadcast.(Float64, Z)
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

	# Compute MCMC samples
	θ = Folds.map(1:K) do k
		 @info "MCMC sampling parameters for data set $k out of $K"
		 Dₖ = D[D_pointer[k]]
		 MCMC(Z[k], θ₀[k], Dₖ, Ω)
	end

	# Convert from vector of vectors to matrix
	θ = reduce(hcat, θ)

	return θ
end

function MCMC(Z::M, θ₀::V, D, Ω; nMH = 7000, burn = 2000, thin = 10) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}

	# initialise MCMC chain
	θ = Array{typeof(θ₀)}(undef, nMH)
	θ[1] = θ₀

	# prior support
	Ω = [Ω...] # convert to array since broadcasting over dictionaries and NamedTuples is reserved

	# compute initial likelihood
	ℓ_current = logmvnorm(θ[1], Z, D)
	n_accept = 0
	sd_propose = 0.1

	for i in 2:nMH

		## Propose
		# multivariate Gaussian
		θ_prop = θ[i-1] .+ sd_propose .* randn(2)

		## Accept/Reject
		if !all(θ_prop .∈ support.(Ω))
		  α = 0
		else
		  ℓ_new = logmvnorm(θ_prop, Z, D)
		  α = exp(ℓ_new - ℓ_current)
	  	end

		if rand(1)[1] < α
		  # Accept
		  θ[i] = θ_prop
		  ℓ_current = ℓ_new
		  n_accept += 1
		else
		  ## Reject
		  θ[i] = θ[i-1]
	  	end

		## Monitor acceptance rate
		acc_ratio = n_accept / i
		if i % 1000 == 0 println("Sample $i: Acceptance rate: $acc_ratio") end
		## If within the burn-in period, adapt acceptance rate
		if (i < burn) & (i % 100 == 0)
		  if acc_ratio < 0.15
			## Decrease proposal variance
			sd_propose /= 1.1
		  else acc_ratio > 0.4
			## Increase proposal variance
			sd_propose *= 1.1
		  end
	  	end
	end

	# remove burn-in samples and thin the chain
	θ = θ[(burn+1):end]
	θ = θ[1:thin:end]

	# compute marginal medians
	θ = reduce(hcat, θ)
	θ = median(θ; dims = 2)
	θ = vec(θ)

	return θ
end


function covariancematrix(D; τ, ρ, ν, σ²)
	# Exploit symmetry of D to minimise the number of computations
    Σ = matern.(UpperTriangular(D), ρ, ν, σ²)
	Σ[diagind(Σ)] .+= τ^2
    return Σ
end

function logmvnorm(θ, Z, D)
	ν  = one(eltype(θ)) # smoothness fixed to 1
	p  = length(θ)
	σ  = p > 2 ? θ[3] : one(eltype(θ))
	σ² = σ^2
	Σ = covariancematrix(D; τ = θ[1], ρ = θ[2], ν = ν, σ² = σ²)
	ℓ = gaussiandensity(Z, Σ; logdensity = true)
	return ℓ
end
