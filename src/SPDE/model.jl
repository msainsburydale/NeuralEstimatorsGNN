using NeuralEstimators
import NeuralEstimators: simulate
using NeuralEstimatorsGNN
import NeuralEstimatorsGNN: Parameters
using Distances: pairwise, Euclidean
using Distributions: Uniform
using LinearAlgebra
using RCall

Ω = (
	ρ = Uniform(0.05, 0.3),
	ν = Uniform(0.5, 1.5)
)
parameter_names = String.(collect(keys(Ω)))

ξ = (
	Ω = Ω,
	p = length(Ω),
	n = 250,
	parameter_names = parameter_names,
	ρ_idx = findfirst(parameter_names .== "ρ"),
	ν_idx = findfirst(parameter_names .== "ν"),
	σ = 1.0,                 # marginal variance to use if σ is not included in Ω
	r = 0.15f0,              # cutoff distance used to define the neighbourhood of each node
	invtransform = x -> x^3  # inverse of variance-stabilising transformation
)


function Parameters(K::Integer, ξ; J::Integer = 1)

	# The simulation function is not currently designed to exploit J: simply
	# return a parameter object with N = KJ total parameters.
	θ = [rand(ϑ, K*J) for ϑ in ξ.Ω]
	θ = permutedims(hcat(θ...))

	# locations are stored in ξ
	locs = ξ.S
	if typeof(locs) <: Matrix
		loc_pointer = repeat([1], K*J)
		locs = [locs]
	else
		@assert length(locs) == K
		loc_pointer = repeat(1:K, inner = J)
	end


	Parameters(θ, locs, loc_pointer)
end

function simulate(parameters::Parameters, m::R) where {R <: AbstractRange{I}} where I <: Integer

	@assert m == 1:1 "The SPDE simulation function in R only caters for m=1 independent replicate"

	K = size(parameters, 2)
	m̃ = rand(m, K)
	θ = parameters.θ
	ρ = θ[1, :]
	ν = θ[2, :]

	# The names "chols" and "chol_pointer" are just relics from the previous
	# models, since these fields normally store the Cholesky factors.
	locs        = parameters.chols
	loc_pointer = parameters.chol_pointer

	# Providing R with multiple parameter vectors
	loc = [locs[loc_pointer[k]][:, :] for k ∈ 1:K] # need length(loc) == K
	z = simulateSPDE(loc, ρ, ν)
	z = broadcast.(Float32, z)

	return z
end
simulate(parameters::Parameters, m::Integer) = simulate(parameters, range(m, m))
simulate(parameters::Parameters) = stackarrays(simulate(parameters, 1))


# Load the R function for simulation
R"""
source('src/SPDE/simulate.R')
"""

function simulateSPDE(loc::V, ρ, ν, variance_stabilise = true) where {V <: AbstractVector{M}} where {M <: AbstractMatrix{T}} where {T}

	@assert length(loc) == length(ρ) == length(ν) # == length(m)

	range  = ρ
	smooth = ν
	@rput range
	@rput smooth
	@rput loc
	R"""
	# Note that vectors of matrices in Julia are converted to a list of matrices
	# in R, so we need to index loc using double brackets
	K = length(loc)
	z = mclapply(1:K, function(k) {
			z = simulate(fem, loc=loc[[k]], range=range[k], smooth=smooth[k])
			z = as.matrix(z)
			z
	})
	"""

	@rget z
	if variance_stabilise z = broadcast.(cbrt, z) end

	R"""
	# remove data from memory to alleviate memory pressure
	rm(z)
	"""

	return z
end


# n = 250
# S = rand(n, 2)
# Z = simulateSPDE([S, S], [0.5, 0.5], [1.2, 1.2])
# θ = Parameters(100, (ξ..., S = S))
# @time simulate(θ, 1);
