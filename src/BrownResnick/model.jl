using RCall # this is needed to interact with the R scripts that define model simulation.

using NeuralEstimators
import NeuralEstimators: simulate
using NeuralEstimatorsGNN
import NeuralEstimatorsGNN: Parameters
using Distances: pairwise, Euclidean
using Distributions: Uniform
using LinearAlgebra

Ω = (
	ρ = Uniform(0.05, 0.3),
	ν = Uniform(0.5, 1.5)
)
parameter_names = String.(collect(keys(Ω)))

ξ = (
	Ω = Ω,
	p = length(Ω),
	n = 100,
	parameter_names = parameter_names,
	ρ_idx = findfirst(parameter_names .== "ρ"),
	ν_idx = findfirst(parameter_names .== "ν"),
	σ = 1.0,                # marginal variance to use if σ is not included in Ω
	r = 0.15f0,             # cutoff distance used to define the neighbourhood of each node
	invtransform = identity # inverse of variance-stabilising transformation
)

function simulate(parameters::Parameters, m::R) where {R <: AbstractRange{I}} where I <: Integer

	K = size(parameters, 2)
	m̃ = rand(m, K)
	θ = parameters.θ
	ρ = θ[1, :]
	ν = θ[2, :]

	# NB chols and chol_pointer are just relics from the previous models, since
	# these fields normally store the Cholesky factors.
	locs        = parameters.chols
	loc_pointer = parameters.chol_pointer

	Z = map(1:K) do k

		loc = locs[loc_pointer[k]][:, :]
		z = simulatebrownresnick(loc, ρ[k], ν[k], m̃[k])
		z = Float32.(z)
		z
	end

	return Z
end
simulate(parameters::Parameters, m::Integer) = simulate(parameters, range(m, m))
simulate(parameters::Parameters) = stackarrays(simulate(parameters, 1))


function Parameters(K::Integer, ξ; J::Integer = 1)

	# The simulation function is not currently set up to exploit J: simply
	# return a parameter object with N = KJ total parameters.
	θ = [rand(ϑ, K*J) for ϑ in ξ.Ω]
	θ = permutedims(hcat(θ...))

	# # the locations are stored in ξ
	# # note that this assumes that we have only a single set of locations
	# locs = [ξ.S]
	# loc_pointer = repeat([1], K*J)

	# the locations are stored in ξ
	# note that this assumes that we have only a single set of locations
	locs = ξ.S
	loc_pointer = repeat([1], K*J)

	Parameters(θ, locs, loc_pointer) # pointer here implies only one set of
end


##### Load the R functions for simulation and define the variogram function
R"""
source('src/BrownResnick/simulation_Dombry_et_al.R')

vario <- function(x){
  range <- par_true[1]
  smooth <- par_true[2]
  (sqrt(sum(x^2))/range)^smooth ## power variogram (h/range)^smooth
}
"""

function simulatebrownresnick(loc, ρ, ν, m = 1)
	par_true = [ρ, ν]
	@rput par_true
	@rput m
	@rput loc
	R"""
	z = simu_extrfcts(model='brownresnick',m=m,coord=loc,vario=vario)$res
	z = t(z)
	"""
	@rget z
	return z
end


# Z = simulatebrownresnick(ξ.S, 0.5, 1.2, 30)
# tmp = Parameters(100, ξ)
# @time simulate(tmp, 30);
