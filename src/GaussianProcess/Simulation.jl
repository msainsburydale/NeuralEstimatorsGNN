using NeuralEstimators
import NeuralEstimators: simulate
using NeuralEstimatorsEM
import NeuralEstimatorsEM: simulateconditional
using Folds
using LinearAlgebra
using Statistics
using Test

# ---- Marginal simulation ----

"""
	simulate(parameters::Parameters)
	simulate(parameters::Parameters, m::Integer)
	simulate(parameters::Parameters, m::R) where {R <: AbstractRange{I}} where I <: Integer
Simulates `m` fields from a Gaussian process for each of the given covariance `parameters`.
If `m` is not provided, a single field is simulated for each parameter
configuration, and the return type is an array with the last dimension
corresponding to the parameters. If `m` is provided, `m` fields
are simulated for each parameter configuration, and the return type is a vector
of arrays equal in length to the number of parameter configurations, and with
the fourth dimension of the array containing the field replicates.
This function assumes that the nugget standard deviation parameters is stored in the first row of `parameters.θ`.
"""
function simulate(parameters::Parameters, m::R) where {R <: AbstractRange{I}} where I <: Integer

	P = size(parameters, 2)
	m̃ = rand(m, P)

	σ        = Float64.(parameters.θ[1, :])
	chols    = parameters.chols
	chol_idx = parameters.chol_idx

	# Folds.map() is type unstable. I've opened an issue with the package, but
	# have not received a response yet. To improve efficiency, I may need to use
	# an alternative parallel mapping function.
	Z = Folds.map(1:P) do i
		L = view(chols, :, :, chol_idx[i])
		z = simulategaussianprocess(L, σ[i], m̃[i])
		z = Float32.(z)
		z
	end
	n = size(chols, 1)
	Z = reshape.(Z, isqrt(n), isqrt(n), 1, :) # assumes a square domain
	return Z
end
simulate(parameters::Parameters, m::Integer) = simulate(parameters, range(m, m))
simulate(parameters::Parameters) = stackarrays(simulate(parameters, 1))


# ---- Conditional simulation ----

function simulateconditionalgaussianprocess(Zₒ::A, θ, ξ; nsims::Integer) where {A <: AbstractArray{Union{Missing, Float32}, N}} where N

	@assert size(Zₒ)[end] == 1

	θ  = Float64.(θ)
    σₑ = θ[1]
    ρ  = θ[2]

	# Determine if we are estimating ν
	estimate_ν = "ν" ∈ ξ.parameter_names
	ν = estimate_ν ? θ[3] : ξ.ν

	# Compute indices of the observed and missing data
	Jᵤ = findall(x -> ismissing(x), Zₒ)  # cartesian indices of missing data
	Jₒ = findall(x -> !ismissing(x), Zₒ) # cartesian indices of observed data
	Iᵤ = LinearIndices(Zₒ)[Jᵤ]           # linear indices of missing data
	Iₒ = LinearIndices(Zₒ)[Jₒ]           # linear indices of observed data
	nᵤ = length(Iᵤ)
	nₒ = length(Iₒ)

	# Distance matrices needed for various covariance matrices
	D   = ξ.D # distance matrix for all locations in the grid
	Dᵤᵤ = D[Iᵤ, Iᵤ]
	Dₒₒ = D[Iₒ, Iₒ]
	Dₒᵤ = D[Iₒ, Iᵤ]

	# Compute the covariance matrices for the latent process Y(⋅)
	Cᵤᵤ = matern.(Dᵤᵤ, ρ, ν)
	Cₒₒ = matern.(Dₒₒ, ρ, ν)
	Cₒᵤ = matern.(Dₒᵤ, ρ, ν)

	# Compute the covariance matrices for the data
	Σᵤᵤ = Cᵤᵤ; Σᵤᵤ[diagind(Σᵤᵤ)] .+= σₑ^2
	Σₒₒ = Cₒₒ; Σₒₒ[diagind(Σₒₒ)] .+= σₑ^2
	Σₒᵤ = Cₒᵤ

	# Compute the Cholesky factor of Σₒₒ and solve the lower triangular systems
	Lₒₒ = cholesky(Symmetric(Σₒₒ)).L
	x = Lₒₒ \ Σₒᵤ
	y = Lₒₒ \ Zₒ[Iₒ]

	# Conditonal mean and covariance matrices (omit subscripts for convenience)
	μ = x'y
	Σ = Σᵤᵤ - x'x

	# Simulate from the distribution Zᵤ ∣ Zₒ, θ ∼ N(μ, Σ)
	L = cholesky(Symmetric(Σ)).L
	z = randn(nᵤ, nsims)
	Zᵤ = μ .+ L * z

	# Combine the observed and missing data to form the complete data
	Z = map(1:nsims) do l
		Z = copy(Zₒ)
		Z[Jᵤ] = Zᵤ[:, l]
		Z
	end

	Z = stackarrays(Z, merge = true) |> copy
	Z = convert(Array{Float32, N}, Z)

	return Z
end


# TODO update as per the NMVM code (passing just the vectors rather than a grid)
function simulateconditional(Zₒ::A, θ, ξ; nsims::Integer = 1) where {A <: AbstractArray{Union{Missing, T}, N}} where {T, N}

	# The last dimension of Zₒ contains the replicates: the other dimensions
	# store the response variable.
	colons = ntuple(_ -> (:), ndims(Zₒ) - 1)
	m = size(Zₒ)[end]
	Z = map(1:m) do i
		simulateconditionalgaussianprocess(Zₒ[colons..., i], θ, ξ, nsims = nsims)
	end

	# Convert Z to the correct shape
	Z = stackarrays(Z)
	Z = reshape(Z, size(Zₒ)[1:end-1]..., m)

	return Z
end
