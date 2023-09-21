# -------------------------------------------------------------------
# ---- Experiment: GNNs in the presence of variable sample sizes ----
# -------------------------------------------------------------------

using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true

end
parsed_args = parse_args(arg_table)
quick           = parsed_args["quick"]

model="GP/nuFixed"
m=[1]

M = maximum(m)
using NeuralEstimators
using NeuralEstimatorsGNN
using BenchmarkTools
using DataFrames
using GraphNeuralNetworks
using CSV

include(joinpath(pwd(), "src/$model/model.jl"))
include(joinpath(pwd(), "src/$model/ML.jl"))
include(joinpath(pwd(), "src/architecture.jl"))

path = "intermediates/supplement/variablesamplesize_n1000/$model"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
K_train = 10_000
K_val   = K_train ÷ 10
if quick
	K_train = K_train ÷ 10
	K_val   = K_val   ÷ 10
	K_val   = max(K_val, 10)
end
K_test = K_val

p = ξ.p
n = ξ.n

# The number of epochs used during training: note that early stopping means that
# we never really train for the full amount of epochs
epochs = quick ? 2 : 200

# ---- Estimators ----

seed!(1)
gnn1 = gnnarchitecture(p)
gnn2 = gnnarchitecture(p, propagation = "WeightedGINConv", aggr = +)

# ---- Training ----

J = 3
large_n = 1000

seed!(1)
θ_val   = Parameters(K_val,   ξ, large_n, J = J)
θ_train = Parameters(K_train, ξ, large_n, J = J)
train(gnn1, θ_train, θ_val, simulate, m = M, savepath = path * "/runs_GNN1", epochs = epochs, epochs_per_Z_refresh = 3)
train(gnn2, θ_train, θ_val, simulate, m = M, savepath = path * "/runs_GNN2", epochs = epochs, epochs_per_Z_refresh = 3)

# ---- Load the trained estimators ----

Flux.loadparams!(gnn1,  loadbestweights(path * "/runs_GNN1"))
Flux.loadparams!(gnn2,  loadbestweights(path * "/runs_GNN2"))

# ---- Assess the estimators ----

function assessestimators(θ, Z, ξ)

	println("	Running GNN estimators...")
	assessment = assess(
		[gnn1, gnn2], θ, Z;
		estimator_names = ["GNN1", "GNN2"],
		parameter_names = ξ.parameter_names,
		verbose = false
	)

	println("	Running maximum-likelihood estimator...")
	# Convert Z from a graph to a matrix (required for maximum-likelihood)
	ξ = (ξ..., θ₀ = θ.θ) # initialise the maximum-likelihood to the true parameters
	Z = broadcast(z -> reshape(z.ndata.x, :, 1), Z)
	assessment = merge(
		assessment,
		assess([ML], θ, Z; estimator_names = ["ML"], parameter_names = ξ.parameter_names, use_gpu = false, use_ξ = true, ξ = ξ, verbose=false)
		)

	return assessment
end

function assessestimators(n, ξ, K::Integer)
	println("	Assessing estimators with n = $n...")

	# test set for estimating the risk function
	seed!(1)
	θ = Parameters(K, ξ, n)
	Z = simulate(θ, M)
	# ML estimator requires the locations and distance matrix:
	S = θ.locations
	D = pairwise.(Ref(Euclidean()), S, S, dims = 1)
	ξ = (ξ..., S = S, D = D)

	assessment = assessestimators(θ, Z, ξ)
	assessment.df[:, :n] .= n
	assessment.runtime[:, :n] .= n

	return assessment
end


assessment = assessestimators(20, ξ, K_test) # dummy run for compilation
assessment = assessestimators(1000, ξ, K_test)
CSV.write(path * "/estimates.csv", assessment.df)
CSV.write(path * "/runtime.csv", times)
