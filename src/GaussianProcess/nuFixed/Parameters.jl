using NeuralEstimators
using Distances: pairwise, Euclidean
using LinearAlgebra
using Distributions: Uniform

Ω = (
	σ = Uniform(0.1, 1),
	ρ = Uniform(2.0, 10.0)
)
parameter_names = String.(collect(keys(Ω)))
S = expandgrid(1:16, 1:16)
S = Float64.(S)
D = pairwise(Euclidean(), S, S, dims = 1)
ξ = (
	parameter_names = parameter_names,
	Ω = Ω, S = S, D = D, p = length(Ω),
	ρ_idx = findfirst(parameter_names .== "ρ"),
	invtransform = identity,
	ν = 1
)
