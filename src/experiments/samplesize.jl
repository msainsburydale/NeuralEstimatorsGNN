# -------------------------------------------------------------------
# ---- Experiment: Applying GNNs to different graph structures ----
# -------------------------------------------------------------------

# Applying GNNs to different graph structures.
# Here we compare a GNN trained under a specific set of irregular locations S,
# and a GNN trained with a variable set of irregular locations {Sₖ : k = 1, …, K},
# where K is the number of unique parameter vectors in the training set.
# We assess the estimators with respect to the set of specific irregular locations, S.
# The purpose of this experiment is to determine whether optimal inference
# requires the neural estimator to be trained specifically under the spatial
# locations of the given data set, or if a general estimator can be used that
# is close to optimal irrespective of the configuration of the spatial locations.

using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--model"
		help = "A relative path to the folder of the assumed model; this folder should contain scripts for defining the parameter configurations in Julia and for data simulation."
		arg_type = String
		required = true
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true
	"--m"
		help = "The sample size to use during training. If multiple samples sizes are given as a vector, multiple neural estimators will be trained."
		arg_type = String
end
parsed_args = parse_args(arg_table)
model           = parsed_args["model"]
quick           = parsed_args["quick"]
m = let expr = Meta.parse(parsed_args["m"])
    @assert expr.head == :vect
    Int.(expr.args)
end

# model="GP/nuFixed"
# quick=true
# m=[1]

M = maximum(m)
using NeuralEstimators
using NeuralEstimatorsGNN
using DataFrames
using GraphNeuralNetworks
using CSV

include(joinpath(pwd(), "src/$model/model.jl"))
include(joinpath(pwd(), "src/$model/MAP.jl"))
include(joinpath(pwd(), "src/architecture.jl"))

path = "intermediates/experiments/samplesize/$model"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
K_train = 10_000
K_val   = K_train ÷ 5
if quick
	K_train = K_train ÷ 100
	K_val   = K_val   ÷ 100
end
K_test = K_val

p = ξ.p
n = size(ξ.D, 1)
ϵ = ξ.ϵ

# The number of epochs used during training: note that early stopping means that
# we never really train for the full amount of epochs
epochs = quick ? 2 : 1000

# ---- Estimators ----

seed!(1)
gnn1 = gnnarchitecture(p)
gnn2 = gnnarchitecture(p)
gnn3 = gnnarchitecture(p)

# ---- Training ----

# GNN estimator trained with a fixed small n
seed!(1)
θ_val,   Z_val   = variableirregularsetup(ξ, 30, K = K_val, m = M, ϵ = ϵ)
θ_train, Z_train = variableirregularsetup(ξ, 30, K = K_train, m = M, ϵ = ϵ)
train(gnn1, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_GNN1", epochs = epochs)

# GNN estimator trained with a fixed large n
seed!(1)
θ_val,   Z_val   = variableirregularsetup(ξ, 300, K = K_val, m = M, ϵ = ϵ)
θ_train, Z_train = variableirregularsetup(ξ, 300, K = K_train, m = M, ϵ = ϵ)
train(gnn2, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_GNN2", epochs = epochs)

# GNN estimator trained with a range of n
seed!(1)
θ_val,   Z_val   = variableirregularsetup(ξ, 30:300, K = K_val, m = M, ϵ = ϵ)
θ_train, Z_train = variableirregularsetup(ξ, 30:300, K = K_train, m = M, ϵ = ϵ)
train(gnn3, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_GNN3", epochs = epochs)

# ---- Load the trained estimators ----

Flux.loadparams!(gnn1,  loadbestweights(path * "/runs_GNN1"))
Flux.loadparams!(gnn2,  loadbestweights(path * "/runs_GNN2"))
Flux.loadparams!(gnn3,  loadbestweights(path * "/runs_GNN3"))

# ---- Assess the estimators ----

function assessestimators(θ, Z, ξ)
	assessment = assess(
		[gnn1, gnn2, gnn3], θ, Z;
		estimator_names = ["GNN1", "GNN2", "GNN3"],
		parameter_names = ξ.parameter_names,
		verbose = false
	)
	# ξ = (ξ..., θ₀ = θ.θ)
	# assessment = merge(assessment, assess([MAP], θ, Z; estimator_names = ["MAP"], parameter_names = ξ.parameter_names, use_gpu = false, use_ξ = true, ξ = ξ))
	return assessment
end

function assessestimators(n, ξ, K::Integer)
	println("	Assessing estimators with n = $n...")
	# test set for estimating the risk function
	seed!(1)
	θ, Z = variableirregularsetup(ξ, n, K = K, m = M, ϵ = ϵ)
	assessment = assessestimators(θ, Z, ξ)
	assessment.df[:, :n] .= n

	return assessment
end

assessment = [assessestimators(n, ξ, K_test) for n ∈ [30, 60, 100, 200, 300, 500, 750, 1000]]
assessment = merge(assessment...)
CSV.write(path * "/estimates_test.csv", assessment.df)
