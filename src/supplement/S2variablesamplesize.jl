# -----------------------------------------------------
# ---- Experiment: GNNs with variable sample sizes ----
# -----------------------------------------------------

using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true

end
parsed_args = parse_args(arg_table)
quick       = parsed_args["quick"]

model= joinpath("GP", "nuSigmaFixed")
m=1
using NeuralEstimators
using NeuralEstimatorsGNN
using BenchmarkTools
using BSON: @load
using DataFrames
using GraphNeuralNetworks
using CSV
using CUDA

include(joinpath(pwd(), "src/$model/model.jl"))
include(joinpath(pwd(), "src/$model/ML.jl"))
include(joinpath(pwd(), "src/architecture.jl"))

path = "intermediates/supplement/variablesamplesize"
if !isdir(path) mkpath(path) end

# Size of the training and test sets
K_train = quick ? 1000 : 5000
K_test  = quick ? 100 : 1000
J = 3 # number of simulated data sets for each parameter configuration

p = ξ.p
n = ξ.n

# ---- Estimators ----

seed!(1)
gnn1 = gnnarchitecture(p)
gnn2 = deepcopy(gnn1)
gnn3 = deepcopy(gnn1)

# ---- Training ----

# The number of epochs used during training: note that early stopping means that
# we never train for the full amount of epochs
epochs = quick ? 2 : 200
cluster_process = false
small_n = 100
large_n = 1000

@info "Training the GNN estimator with n = $(small_n)"
seed!(1)
θ_val   = Parameters(K_train ÷ 5,   ξ, small_n, J = J, cluster_process = cluster_process)
θ_train = Parameters(K_train, ξ, small_n, J = J, cluster_process = cluster_process)
train(gnn1, θ_train, θ_val, simulate, m = m, savepath = path * "/runs_GNN1", epochs = epochs, epochs_per_Z_refresh = 3, stopping_epochs = 5)

@info "Training the GNN estimator with n = $(large_n)"
seed!(1)
θ_val   = Parameters(K_train ÷ 5,   ξ, large_n, J = J, cluster_process = cluster_process)
θ_train = Parameters(K_train, ξ, large_n, J = J, cluster_process = cluster_process)
train(gnn2, θ_train, θ_val, simulate, m = m, savepath = path * "/runs_GNN2", epochs = epochs, epochs_per_Z_refresh = 3, stopping_epochs = 5)

@info "Training the GNN estimator with n ∈ $(small_n:large_n)"
seed!(1)
θ_val   = Parameters(K_train ÷ 5,   ξ, small_n:large_n, J = J, cluster_process = cluster_process)
θ_train = Parameters(K_train, ξ, small_n:large_n, J = J, cluster_process = cluster_process)
train(gnn3, θ_train, θ_val, simulate, m = m, savepath = path * "/runs_GNN3", epochs = epochs, epochs_per_Z_refresh = 3, stopping_epochs = 5)

# ---- Load the trained estimators ----

loadpath  = joinpath(path, "runs_GNN1", "best_network.bson")
@load loadpath model_state
Flux.loadmodel!(gnn1, model_state)

loadpath  = joinpath(path, "runs_GNN2", "best_network.bson")
@load loadpath model_state
Flux.loadmodel!(gnn2, model_state)

loadpath  = joinpath(path, "runs_GNN3", "best_network.bson")
@load loadpath model_state
Flux.loadmodel!(gnn3, model_state)

# ---- Assess the estimators ----

function assessestimators(n, ξ, K::Integer)
	println("	Assessing estimators with n = $n...")

	# test set for estimating the risk function
	seed!(1)
	θ = Parameters(K, ξ, n; cluster_process = false, J = 5)
	Z = simulate(θ, m)

	assessment = assess(
		[gnn1, gnn2, gnn3], θ, Z;
		estimator_names = ["GNN1", "GNN2", "GNN3"],
		parameter_names = ξ.parameter_names,
		verbose = false
	)
	assessment.df[:, :n] .= n
	assessment.runtime[:, :n] .= n

	return assessment
end

test_n = [30, 60, 100, 200, 350, 500, 750, 1000]
assessment = [assessestimators(n, ξ, K_test) for n ∈ test_n]
assessment = merge(assessment...)
CSV.write(path * "/estimates.csv", assessment.df)