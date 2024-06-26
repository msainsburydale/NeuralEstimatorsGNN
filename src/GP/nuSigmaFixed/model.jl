using Distributions: Uniform
Ω = (
	τ = Uniform(0.0, 1.0),
	ρ = Uniform(0.05, 0.5)
)

# To reduce code repetition, use a single file for all GP models
include(joinpath(pwd(), "src/GP/model.jl"))
