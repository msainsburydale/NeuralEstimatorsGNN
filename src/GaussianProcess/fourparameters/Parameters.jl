using NeuralEstimators
using Distances: pairwise, Euclidean
using LinearAlgebra
using Distributions: Uniform

Ω = (
	τ = Uniform(0.5, 0.55),
	ρ = Uniform(0.01, 0.4),
	ν = Uniform(1.0, 2.5),
	σ = Uniform(0.7, 2.5)
)
parameter_names = String.(collect(keys(Ω)))
ξ = (
	parameter_names = parameter_names,
	Ω = Ω,
	p = length(Ω),
	ρ_idx = findfirst(parameter_names .== "ρ"),
	ν_idx = findfirst(parameter_names .== "ν"),
	σ_idx = findfirst(parameter_names .== "σ"),
	invtransform = identity
)
