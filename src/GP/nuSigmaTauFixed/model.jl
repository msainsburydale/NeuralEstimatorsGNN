using Distributions: Uniform
Ω = (
	ρ = Uniform(0.05, 0.5),
)

# To reduce code repetition, use a single file for all GP models
include(joinpath(pwd(), "src/GP/model.jl"))
