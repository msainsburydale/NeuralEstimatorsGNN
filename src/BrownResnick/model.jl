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
	n = 125,
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

function simulate(parameters::Parameters, m::R) where {R <: AbstractRange{I}} where I <: Integer

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
	z = simulatebrownresnick(loc, ρ, ν, m̃)
	z = broadcast.(Float32, z)

	return z
end
simulate(parameters::Parameters, m::Integer) = simulate(parameters, range(m, m))
simulate(parameters::Parameters) = stackarrays(simulate(parameters, 1))


##### Load the R functions for simulation and define the variogram function
R"""
source('src/BrownResnick/simulation_Dombry_et_al.R')

vario <- function(x, range, smooth){
  (sqrt(sum(x^2))/range)^smooth ## power variogram (h/range)^smooth
}
"""

# Providing R with only a single parameter vector at a time
# function simulatebrownresnick(loc, ρ, ν, m, Gumbel = true)
# 	range  = ρ
# 	smooth = ν
# 	@rput range
# 	@rput smooth
# 	@rput m
# 	@rput loc
# 	R"""
# 	z = simu_extrfcts(model='brownresnick',m=m,coord=loc,vario=vario, range=range,smooth=smooth)$res
# 	z = as.matrix(z)
# 	if (m > 1) z = t(z)
# 	"""
# 	@rget z
# 	if Gumbel z = log.(z) end # transform from the unit-Fréchet scale (extremely heavy tailed) to the Gumbel scale
#
# 	R"""
# 	# remove data from memory to alleviate memory pressure
# 	rm(z)
# 	"""
#
# 	return z
# end


# Providing R with multiple parameter vectors
function simulatebrownresnick(loc::V, ρ, ν, m, Gumbel = true) where {V <: AbstractVector{M}} where {M <: AbstractMatrix{T}} where {T}

	@assert length(loc) == length(ρ) == length(ν) == length(m)

	range  = ρ
	smooth = ν
	@rput range
	@rput smooth
	@rput m
	@rput loc
	R"""
	# Note that vectors of matrices in Julia are converted to a list of matrices
	# in R, so we need to index loc using double brackets
	K = length(loc)
	z = mclapply(1:K, function(k) {
			z = simu_extrfcts(model='brownresnick',m=m[k],coord=loc[[k]],vario=vario, range=range[k], smooth=smooth[k])$res
			z = as.matrix(z)
			if (m > 1) z = t(z)
			z
	})
	"""

	@rget z
	if Gumbel z = broadcast.(log, z) end # transform from the unit-Fréchet scale (extremely heavy tailed) to the Gumbel scale

	R"""
	# remove data from memory to alleviate memory pressure
	rm(z)
	"""

	return z
end


# n = 250
# m = 30
# S = rand(n, 2)
# Z = simulatebrownresnick(S, 0.5, 1.2, m)
# Z = simulatebrownresnick([S, S], [0.5, 0.5], [1.2, 1.2], [m, m])
# θ = Parameters(100, (ξ..., S = S))
# @time simulate(θ, m);

# no parallel:   73.648734 seconds (1.37 M allocations: 82.693 MiB, 0.01% gc time, 0.18% compilation time)
# with parallel: 37.257721 seconds (980.92 k allocations: 61.668 MiB, 0.14% gc time, 0.13% compilation time)
