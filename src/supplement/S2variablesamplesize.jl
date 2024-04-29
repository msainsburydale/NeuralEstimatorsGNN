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
using DataFrames
using GraphNeuralNetworks
using CSV

include(joinpath(pwd(), "src/$model/model.jl"))
include(joinpath(pwd(), "src/$model/ML.jl"))
include(joinpath(pwd(), "src/architecture.jl"))

path = "intermediates/supplement/variablesamplesize"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
K_train = 10_000
K_val   = K_train ÷ 10
if quick
	K_train = K_train ÷ 100
	K_val   = K_val   ÷ 100
	K_val   = max(K_val, 100)
end
K_test = K_val

p = ξ.p
n = ξ.n

# The number of epochs used during training: note that early stopping means that
# we never train for the full amount of epochs
epochs = quick ? 2 : 200

# ---- Estimators ----

seed!(1)
gnn1 = gnnarchitecture(p)
gnn2 = deepcopy(gnn1)
gnn3 = deepcopy(gnn1)

# ---- Training ----

J = 3
small_n = 30
large_n = 1000

@info "Training the GNN estimator with n = $(small_n)"
seed!(1)
θ_val   = Parameters(K_val,   ξ, small_n, J = J, cluster_process = false)
θ_train = Parameters(K_train, ξ, small_n, J = J, cluster_process = false)
train(gnn1, θ_train, θ_val, simulate, m = m, savepath = path * "/runs_GNN1", epochs = epochs, epochs_per_Z_refresh = 3, stopping_epochs = 5)

@info "Training the GNN estimator with n = $(large_n)"
seed!(1)
θ_val   = Parameters(K_val,   ξ, large_n, J = J, cluster_process = false)
θ_train = Parameters(K_train, ξ, large_n, J = J, cluster_process = false)
train(gnn2, θ_train, θ_val, simulate, m = m, savepath = path * "/runs_GNN2", epochs = epochs, epochs_per_Z_refresh = 3, stopping_epochs = 5)

@info "Training the GNN estimator with n ∈ $(small_n:large_n)"
seed!(1)
θ_val   = Parameters(K_val,   ξ, small_n:large_n, J = J, cluster_process = false)
θ_train = Parameters(K_train, ξ, small_n:large_n, J = J, cluster_process = false)
train(gnn3, θ_train, θ_val, simulate, m = m, savepath = path * "/runs_GNN3", epochs = epochs, epochs_per_Z_refresh = 3, stopping_epochs = 5)

# ---- Load the trained estimators ----

Flux.loadparams!(gnn1,  loadbestweights(path * "/runs_GNN1"))
Flux.loadparams!(gnn2,  loadbestweights(path * "/runs_GNN2"))
Flux.loadparams!(gnn3,  loadbestweights(path * "/runs_GNN3"))

# ---- Assess the estimators ----

function assessestimators(θ, Z, ξ)

	println("	Running GNN estimators...")
	assessment = assess(
		[gnn1, gnn2, gnn3], θ, Z;
		estimator_names = ["GNN1", "GNN2", "GNN3"],
		parameter_names = ξ.parameter_names,
		verbose = false
	)

	# println("	Running maximum-likelihood estimator...")
	# # Convert Z from a graph to a matrix (required for maximum-likelihood)
	# ξ = (ξ..., θ₀ = θ.θ) # initialise the maximum-likelihood to the true parameters
	# Z = broadcast(z -> reshape(z.ndata.x, :, 1), Z)
	# assessment = merge(
	# 	assessment,
	# 	assess([ML], θ, Z; estimator_names = ["ML"], parameter_names = ξ.parameter_names, use_gpu = false, use_ξ = true, ξ = ξ, verbose=false)
	# 	)

	return assessment
end

function assessestimators(n, ξ, K::Integer)
	println("	Assessing estimators with n = $n...")

	# test set for estimating the risk function
	seed!(1)
	θ = Parameters(K, ξ, n; cluster_process = false)
	Z = simulate(θ, m)
	# ML estimator requires the locations and distance matrix:
	S = θ.locations
	#D = pairwise.(Ref(Euclidean()), S, dims = 1)
	ξ = (ξ..., S = S)#, D = D)

	assessment = assessestimators(θ, Z, ξ)
	assessment.df[:, :n] .= n
	assessment.runtime[:, :n] .= n

	return assessment
end

test_n = [30, 60, 100, 200, 350, 500, 750, 1000, 1500, 2000]
assessment = [assessestimators(n, ξ, K_test) for n ∈ test_n]
assessment = merge(assessment...)
CSV.write(path * "/estimates.csv", assessment.df)

# ----------------------------------------------------------------------------
# ---- Quantile estimator ----
# ----------------------------------------------------------------------------

seed!(1)
v = gnnarchitecture(p; final_activation = identity)
Flux.loadparams!(v, loadbestweights(path * "/runs_GNN3")) # pretrain with point estimator
intervalestimator = IntervalEstimator(v)

train(intervalestimator, θ_train, θ_val, simulate, m = m, savepath = path * "/runs_intervalestimator", epochs = epochs, epochs_per_Z_refresh = 3, stopping_epochs = 5)
Flux.loadparams!(intervalestimator,  loadbestweights(path * "/runs_intervalestimator"))

function assessestimators(n, ξ, K::Integer)
	println("	Assessing interval estimator with n = $n...")

	# test parameter vectors for estimating the risk function
	seed!(1)
	θ = Parameters(K, ξ, n, cluster_process = false)
	Z = simulate(θ, m)

	assessment = assess(
		intervalestimator, θ, Z;
		estimator_name = "intervalestimator",
		parameter_names = ξ.parameter_names,
		verbose = false
	)

	# Add sample size information
	assessment.df[:, :n] .= n

	return assessment
end

assessment = [assessestimators(n, ξ, 3K_test) for n ∈ test_n]
assessment = merge(assessment...)
CSV.write(joinpath(path, "estimates_interval.csv"), assessment.df)
