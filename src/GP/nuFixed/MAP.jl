# To reduce code repetition, I have used a single file
# for both ν fixed and ν unknown. Here, we simply source that file.
include(joinpath(pwd(), "src/spatial/GaussianProcess/MAP.jl"))
