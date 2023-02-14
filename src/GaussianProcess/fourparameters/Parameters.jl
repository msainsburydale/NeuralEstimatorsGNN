using NeuralEstimators
using Distances: pairwise, Euclidean
using LinearAlgebra
using Distributions: Uniform

Ω = (
	τ = Uniform(0.1, 1.0),
	ρ = Uniform(0.1, 0.7),
	ν = Uniform(0.5, 2.0),
	σ = Uniform(0.1, 1.0)
)
parameter_names = String.(collect(keys(Ω)))
n = 900 # number of locations in each field
ξ = (
	parameter_names = parameter_names,
	Ω = Ω,
	p = length(Ω),
	n = n,
	ρ_idx = findfirst(parameter_names .== "ρ"),
	ν_idx = findfirst(parameter_names .== "ν"),
	σ_idx = findfirst(parameter_names .== "σ"),
	invtransform = identity
)
