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
	σ = 1.0,                # marginal variance to use if σ is not included in Ω
	r = 0.15f0,             # cutoff distance used to define the neighbourhood of each node
	invtransform = exp      # inverse of variance-stabilising transformation
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


# It is faster to do all of the simulation in R by passing a pxK matrix
# of parameters and a K-vector of m. However this is complicated slightly by
# the fact that we do not necessarily have K unique spatial configurations
# (in the case that J > 1). Further, we may run into memory problems if we store
# the full data set in both R and Julia.

function simulate(parameters::Parameters, m::R; exact::Bool = false) where {R <: AbstractRange{I}} where I <: Integer

	K = size(parameters, 2)
	m̃ = rand(m, K)
	θ = parameters.θ
	ρ = θ[1, :]
	ν = θ[2, :]

	# The names "chols" and "chol_pointer" are just relics from the previous
	# models, since these fields normally store the Cholesky factors.
	locs        = parameters.chols
	loc_pointer = parameters.chol_pointer

	# Providing R with only a single parameter vector at a time
	# z = map(1:K) do k
	# 	loc = locs[loc_pointer[k]][:, :]
	# 	z = simulatebrownresnick(loc, ρ[k], ν[k], m̃[k])
	# 	z = Float32.(z)
	# 	z
	# end

	# Providing R with multiple parameter vectors
	loc = [locs[loc_pointer[k]][:, :] for k ∈ 1:K] # need length(loc) == K
	z = simulatebrownresnick(loc, ρ, ν, m̃, exact = exact)
	z = broadcast.(Float32, z)

	return z
end
simulate(parameters::Parameters, m::Integer; exact::Bool = false) = simulate(parameters, range(m, m); exact = exact)
simulate(parameters::Parameters; exact::Bool = false) = stackarrays(simulate(parameters, 1; exact = exact))




##### Load the R functions for simulation and define the variogram function
# R"""
# source('src/BrownResnick/simulate.R')
# """

R"""
suppressMessages({
library("parallel")
library("SpatialExtremes")
})
"""

# Providing R with only a single parameter vector at a time
function simulatebrownresnick(loc, ρ, ν, m; Gumbel::Bool = true, exact::Bool = false)
	range  = ρ
	smooth = ν
	@rput range
	@rput smooth
	@rput m
	@rput loc
  	@rput exact
	# R"""
	# z = simulateBR(m=m, coord=loc, vario=vario, range=range, smooth=smooth, exact=exact)
	# """
	R"""
	z = rmaxstab(n=m, coord = coord, range = range, smooth = smooth, cov.mod = "brown", control = list(method = "exact"))
	z = t(z)
	"""
	@rget z
	if Gumbel z = log.(z) end # transform from the unit-Fréchet scale (extremely heavy tailed) to the Gumbel scale

	R"""
	# remove data from memory to alleviate memory pressure
	rm(z)
	"""

	return z
end


# Providing R with multiple parameter vectors and locations
function simulatebrownresnick(loc::V, ρ, ν, m; Gumbel::Bool = true, exact::Bool = false) where {V <: AbstractVector{M}} where {M <: AbstractMatrix{T}} where {T}

	@assert length(loc) == length(ρ) == length(ν) == length(m)

	range  = ρ
	smooth = ν
	@rput range
	@rput smooth
	@rput m
	@rput loc
	@rput exact
	R"""
	## Note that vectors of matrices in Julia are converted to a list of matrices
	## in R, so we need to index loc using double brackets
	K = length(loc)

	z = mclapply(1:K, function(k) {

		# simulateBR(m=m[k], coord=loc[[k]], vario=vario, range=range[k], smooth=smooth[k], exact=exact)

		z = rmaxstab(n=m[k], coord=loc[[k]], range=range[k], smooth=smooth[k], cov.mod = "brown", control = list(method = "exact"))
		z = t(z)
	})
	"""

	@rget z
	if Gumbel z = broadcast.(log, z) end # transform from the unit-Fréchet scale (extremely heavy tailed) to the Gumbel scale

	R"""
	## Remove data from memory in R to alleviate memory pressure
	rm(z)
	"""

	return z
end


# n = 250
# m = 30
# loc = rand(n, 2)
# ρ = 0.2
# ν = 0.7
# simulatebrownresnick(loc, ρ, ν, m)
# @time simulatebrownresnick([loc, loc], [ρ, ρ], [ν, ν], [m, m]);
# @time simulatebrownresnick([loc, loc], [ρ, ρ], [ν, ν], [m, m], exact = true);
# θ = Parameters(10, (ξ..., S = loc))
# @time simulate(θ, m, exact=false);
# @time simulate(θ, m, exact=true);
