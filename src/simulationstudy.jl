using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--model"
		help = "A relative path to the folder of the assumed model; this folder should contain scripts for defining the parameter configurations in Julia and for data simulation."
		arg_type = String
		required = true
	"--skip_training"
		help = "A flag controlling whether or not we should skip training the estimators: useful for running the assessments without retraining the estimators."
		action = :store_true
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true
	"--m"
		help = "The sample size to use during training. If multiple samples sizes are given as a vector, multiple neural estimators will be trained."
		arg_type = String
end
parsed_args = parse_args(arg_table)

model         = parsed_args["model"]
skip_training = parsed_args["skip_training"]
quick         = parsed_args["quick"]
m = let expr = Meta.parse(parsed_args["m"])
    @assert expr.head == :vect
    Int.(expr.args)
end

M = maximum(m)
using NeuralEstimators
using NeuralEstimatorsGNN
using NamedArrays
using BenchmarkTools
using CSV
using DataFrames
using Distances
using GraphNeuralNetworks

include(joinpath(pwd(), "src/$model/model.jl"))
include(joinpath(pwd(), "src/architecture.jl"))
include(joinpath(pwd(), "src/$model/ML.jl"))
# if model == "GP/nuSigmaFixed" include(joinpath(pwd(), "src/MCMC.jl")) end
p = ξ.p
n = ξ.n

# ML initial estimates (and MCMC initial values) taken to be the prior mean
θ₀ = mean.([ξ.Ω...])
ξ = (ξ..., θ₀ = θ₀)

path = "intermediates/$model"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
K_train = 50_000
K_val   = K_train ÷ 5
if quick
	K_train = K_train ÷ 10
	K_val   = K_val   ÷ 10
end

epochs = quick ? 20 : 200
J = 1

if !skip_training
	@info "Generating training data..."
	seed!(1)
	@info "Sampling parameter vectors used for validation..."
	θ_val = Parameters(K_val, ξ, n, J = J)
	@info "Sampling parameter vectors used for training..."
	θ_train = Parameters(K_train, ξ, n, J = J)
end

# -----------------------------------------------------------------------------
# -------------------------- Point estimator ----------------------------------
# -----------------------------------------------------------------------------

seed!(1)
pointestimator = gnnarchitecture(p)

if !skip_training
	@info "Training the GNN point estimator..."
	trainx(
		pointestimator, θ_train, θ_val, simulate, m,
		savepath = joinpath(path, "runs_GNN"),
		epochs = epochs,
		batchsize = 64,
		epochs_per_Z_refresh = 3
		)
end

# Load the trained estimator
Flux.loadparams!(pointestimator,  loadbestweights(joinpath(path, "runs_GNN_m$M")))

# ---- Run-time assessment ----

# Accurately assess the run-time for a single data set
if isdefined(Main, :ML)

	# Simulate data
	seed!(1)
	S = rand(n, 2)
	D = pairwise(Euclidean(), S, dims = 1)
	ξ = (ξ..., S = S, D = D) # update ξ to contain the new spatial locations and distance matrix (the latter is needed for ML estimation)
	θ = Parameters(1, ξ)
	Z = simulate(θ, M; convert_to_graph = false)

	#ML estimates
	t_ml = @belapsed ML(Z, ξ)

	# GNN estimates
	g = θ.graphs[1]
	Z = spatialgraph(g, Z[1])
	Z = Z |> gpu
	pointestimator = pointestimator|> gpu
	t_gnn = @belapsed pointestimator(Z)

	# Save the runtime
	t = DataFrame(time = [t_gnn, t_ml], estimator = ["GNN", "ML"])
	CSV.write(joinpath(path, "runtime.csv"), t)

end

# ---- Assess the point estimators ----

@info "Assessing point estimators..."

K_test = quick ? 50 : 1000

function assessestimators(θ, Z, ξ)

	# Convert the data to a graph
	g = θ.graphs[1]
	Z_graph = spatialgraph.(Ref(g), Z)

	# Assess the GNN
	assessment = assess(pointestimator, θ, Z_graph; estimator_name = "GNN", parameter_names = ξ.parameter_names)

	# Assess the ML estimator (if it is defined)
	if isdefined(Main, :ML)
		assessment = merge(assessment, assess(ML, θ, Z; estimator_name = "ML", ξ = ξ))
	end

	# Assess the posterior median obtained using MCMC (if it is defined)
	if isdefined(Main, :MCMC)
		assessment = merge(assessment, assess(MCMC, θ, Z; estimator_name = "MCMC", ξ = ξ))
	end

	return assessment
end

function assessestimators(ξ, set::String)

	# Generate spatial locations and construct distance matrix
	S = spatialconfigurations(n, set)
	D = pairwise(Euclidean(), S, S, dims = 1)
	ξ = (ξ..., S = S, D = D) # update ξ to contain the new spatial locations and distance matrix (the latter is needed for ML estimation)

	# test set for estimating the risk function
	θ = Parameters(K_test, ξ)
	Z = simulate(θ, M, convert_to_graph = false)
	assessment = assessestimators(θ, Z, ξ)
	CSV.write(path * "/estimates_test_$set.csv", assessment.df)
	CSV.write(path * "/runtime_test_$set.csv", assessment.runtime)

	# small number of parameters for visualising the sampling distributions
	K_scenarios = 5
	seed!(1)
	θ = Parameters(K_scenarios, ξ)
	J = quick ? 10 : 100
	Z = simulate(θ, M, J, convert_to_graph = false)
	assessment = assessestimators(θ, Z, ξ)
	CSV.write(path * "/estimates_scenarios_$set.csv", assessment.df)
	CSV.write(path * "/runtime_scenarios_$set.csv", assessment.runtime)

	# save spatial fields for plotting
	Z = Z[1:K_scenarios] # only need one field per parameter configuration
	colons  = ntuple(_ -> (:), ndims(Z[1]) - 1)
	z  = broadcast(z -> vec(z[colons..., 1]), Z) # save only the first replicate of each parameter configuration
	z  = vcat(z...)
	z  = broadcast(ξ.invtransform, z)
	d  = prod(size(Z[1])[1:end-1])
	k  = repeat(1:K_scenarios, inner = d)
	s1 = repeat(S[:, 1], K_scenarios)
	s2 = repeat(S[:, 2], K_scenarios)
	df = DataFrame(Z = z, k = k, s1 = s1, s2 = s2)
	CSV.write(path * "/Z_$set.csv", df)

	return assessment
end


"""
	spatialconfigurations(n::Integer, set::String)
Generates spatial configurations of size `n` corresponding to one of
the four types of `set`s as used in Section 3 of the manuscript.

# Examples
```
n = 250
S₁ = spatialconfigurations(n, "uniform")
S₂ = spatialconfigurations(n, "quadrants")
S₃ = spatialconfigurations(n, "mixedsparsity")
S₄ = spatialconfigurations(n, "cup")

using UnicodePlots
[scatterplot(S[:, 1], S[:, 2]) for S ∈ [S₁, S₂, S₃, S₄]]
```
"""
function spatialconfigurations(n::Integer, set::String)

	@assert n > 0
	@assert set ∈ ["uniform", "quadrants", "mixedsparsity", "cup"]

	if set == "uniform"
		S = rand(n, 2)
	elseif set == "quadrants"
		S₁ = 0.5 * rand(n÷2, 2)
		S₂ = 0.5 * rand(n÷2, 2) .+ 0.5
		S  = vcat(S₁, S₂)
	elseif set == "mixedsparsity"
		n_centre = (3 * n) ÷ 4
		n_corner = (n - n_centre) ÷ 4
		S_centre  = 1/3 * rand(n_centre, 2) .+ 1/3
		S_corner1 = 1/3 * rand(n_corner, 2)
		S_corner2 = 1/3 * rand(n_corner, 2); S_corner2[:, 2] .+= 2/3
		S_corner3 = 1/3 * rand(n_corner, 2); S_corner3[:, 1] .+= 2/3
		S_corner4 = 1/3 * rand(n_corner, 2); S_corner4 .+= 2/3
		S = vcat(S_centre, S_corner1, S_corner2, S_corner3, S_corner4)
	elseif set == "cup"
		n_strip2 = n÷3 + n % 3 # ensure that total sample size is n (even if n is not divisible by 3)
		S_strip1 = rand(n÷3, 2);      S_strip1[:, 1] .*= 0.2;
		S_strip2 = rand(n_strip2, 2); S_strip2[:, 1] .*= 0.6; S_strip2[:, 1] .+= 0.2; S_strip2[:, 2] .*= 1/3;
		S_strip3 = rand(n÷3, 2);      S_strip3[:, 1] .*= 0.2; S_strip3[:, 1] .+= 0.8;
		S = vcat(S_strip1, S_strip2, S_strip3)
	end

	return S
end


# Test with respect to a set of uniformly sampled locations
#  .   . . . .
#  . . . . .
#  . . . . . .
#  .   . .   .
#  . . . . . .
#  .   . . .
seed!(1)
assessestimators(ξ, "uniform")

# Test with respect to locations sampled only in the first and third quadrants
#         . . .
#         . . .
#         . . .
#  . . .
#  . . .
#  . . .
seed!(1)
assessestimators(ξ, "quadrants")


# Test with respect to locations with mixed sparsity.
# Divide the domain into 9 cells: have the
# central cell being very densely populated; the corner cells sparse; and the
# side cells empty.
# . .         . .
#   . .       .   .
#               .
#       . . .
#       . . .
#       . . .
# . .         .
#    .        . . .
# .             .
seed!(1)
assessestimators(ξ, "mixedsparsity")


# Test with respect to locations with a cup shape ∪.
# . .           . .
# . .           . .
# . .           . .
# . .           . .
# . .           . .
# . .           . .
# . . . . . . . . .
# . . . . . . . . .
# . . . . . . . . .
#Construct by considering the domain split into three vertical strips
seed!(1)
assessestimators(ξ, "cup")



# -----------------------------------------------------------------------------
# --------------------- Uncertainty quantification ----------------------------
# -----------------------------------------------------------------------------

#TODO uncomment

# @info "Constructing and assessing neural quantile..."
# 
# # Credible-interval estimator:
# seed!(1)
# v = gnnarchitecture(p; final_activation = identity)
# intervalestimator = IntervalEstimator(v)
# 
# if !skip_training
# 	@info "training the GNN quantile estimator for marginal posterior credible intervals..."
# 	trainx(
# 	  intervalestimator, θ_train, θ_val, simulate, m, 
# 	  savepath = joinpath(path, "runs_GNN_CI"), 
# 	  epochs = epochs, 
# 	  batchsize = 64, 
# 	  epochs_per_Z_refresh = 3
# 	  )
# end
# 
# Flux.loadparams!(intervalestimator, loadbestweights(joinpath(path, "runs_GNN_CI_m$M")))
# 
# # Assessment
# seed!(2023)
# K_test = quick ? 100 : 1000
# θ_test = Parameters(K_test, ξ, n, J = 1)
# Z_test = simulate(θ_test, M)
# assessment = assess(
#       intervalestimator, θ_test, Z_test, 
#       estimator_name = "quantile", parameter_names = ξ.parameter_names
#       )
# 
# # Compute and save diagnostics
# cov = coverage(assessment)
# is  = intervalscore(assessment)
# uq_assessment = innerjoin(cov, is, on = [:estimator, :parameter])
# CSV.write(joinpath(path, "uq_assessment.csv"), uq_assessment)