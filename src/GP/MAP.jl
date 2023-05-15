using LinearAlgebra
using NeuralEstimators
using NeuralEstimatorsEM
using Optim
using Folds
using Flux: flatten
using Test

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
	end

	# Number of parameter configurations to estimate
	K = size(θ₀, 2)

	# Convert from matrix to vector of vectors
	θ₀ = [θ₀[:, k] for k ∈ 1:K]

	# Optimise
	θ̂ = Folds.map(Z, θ₀) do Zₖ, θ₀ₖ
		 MAP(Zₖ, θ₀ₖ, ξ, Ω) # this is model specific
	end

	# Convert to matrix
	θ̂ = hcat(θ̂...)

	return θ̂
end



function MAP(Z::M, θ₀::V, ξ, Ω) where {T, R, V <: AbstractArray{T, 1}, M <: AbstractArray{R, 2}}

	# Get the indices of the observed data. This is also used to determine if we
	# have different patterns of missingness in the data.
	m   = size(Z, 2)
	idx = [findall(x -> !ismissing(x), vec(Z[:, i])) for i ∈ 1:m]

	# Drop the missing observations from Z, and convert the eltype from Union{Missing, R} to R
	Z = [[Z[idx[i], i]...] for i ∈ 1:m]

	# If the missingness patterns vary, Z needs to be kept as an array of arrays;
	# otherwise, we can store it as a matrix.
	fixed_pattern = constpattern(idx)
	if fixed_pattern
		Z = hcat(Z...)
	end

	# If it is present, the distance matrix D also needs to be modified, since
	# D contains the distances for all locations, but we only want distances for
	# the observed locations. If the missingness patterns vary, D needs to be an
	# array of matrices. Otherwise, we need only a single distance matrix.
	# Note that we deal with the distance matrix here, rather than in nll(),
	# since it is more efficient (we don't need to do it at each iteration) and
	# it reduces code repetition for the models that use distance matrices.
	if haskey(ξ, :D)
		D = ξ.D
		if fixed_pattern
			D = D[idx[1], idx[1]]
		else
			D = [D[idx[i], idx[i]] for i ∈ 1:m]
		end
		ξ = merge(ξ, (D = D,))
	end

	# Closure that will be minimised
	loss(θ) = nll(θ, Z, ξ, Ω)

	# Estimate the parameters
	θ̂ = optimize(loss, θ₀, NelderMead()) |> Optim.minimizer

	# During optimisation, we constrained the parameters using the scaled-logistic
	# function; here, we convert to the orginal scale
	θ̂ = scaledlogistic.(θ̂, Ω)

	return θ̂
end


function covariancematrix(D; τ, ρ, ν)
	# Exploit symmetry of D to minimise the number of computations
    Σ = matern.(UpperTriangular(D), ρ, ν)
	Σ[diagind(Σ)] .+= τ^2
    return Σ
end

"""
Negative log-likelihood function to be minimised using Optim. If length(θ) > 2,
the smoothness parameter, ν, is estimated; otherwise, it is fixed to 1.
"""
function nll(θ, Z, ξ, Ω)

	# Constrain the estimates to be within the prior support
	θ = scaledlogistic.(θ, Ω)
	p = length(θ)
	ν = p > 2 ? θ[3] : one(eltype(θ))

	# compute the log-likelihood function
	ℓ = ll(θ, ν, Z, ξ.D)

	return -ℓ
end

# This method can be used for Z <: AbstractVector and for Z <: AbstractMatrix
function ll(θ, ν, Z, D)
	Σ = covariancematrix(D, τ = θ[1], ρ = θ[2], ν = ν)
	ℓ = gaussiandensity(Z, Σ; logdensity = true)
	return ℓ
end

# This method is for the case that the spatial locations vary between replicates,
# so D is assumed to be an array of matrices.
function ll(θ, ν, Z::V, D) where {T, S <: AbstractVector{T}, V <: AbstractVector{S}}
	m = length(Z)
	ℓ = [ll(θ, ν, vec(Z[i]), ξ.D[i]) for i ∈ 1:m]
	ℓ = sum(ℓ)
	return ℓ
end
