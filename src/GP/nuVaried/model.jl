using Distributions: Uniform
Ω = (
	τ = Uniform(0.1, 1.0),
	ρ = Uniform(0.05, 0.3),
	ν = Uniform(0.5, 1.5)
)

# To reduce code repetition, I have used a file for both ν fixed and ν unknown.
# Here, we simply source that file. Note that we still include a file within
# each subfolder for consistency with the assumed structure of the repo.
include(joinpath(pwd(), "src/GP/model.jl"))
