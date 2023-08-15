# -------------------------------------------------------------------
# ---- Experiment: GNNs in the presence of varibale sample sizes ----
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
neighbours = "radius"

M = maximum(m)
using NeuralEstimators
using NeuralEstimatorsGNN
using DataFrames
using GraphNeuralNetworks
using CSV

include(joinpath(pwd(), "src/$model/model.jl"))
include(joinpath(pwd(), "src/$model/MAP.jl"))
include(joinpath(pwd(), "src/architecture.jl"))

path = "intermediates/experiments/samplesize/$model/$neighbours"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
K_train = 30_000
K_val   = K_train ÷ 10
if quick
	K_train = K_train ÷ 100
	K_val   = K_val   ÷ 100
end
K_test = K_val

p = ξ.p
n = size(ξ.D, 1)

# For uniformly sampled locations on a unit grid, the probability that a point
# falls within a circle of radius d is πd². So, on average, we expect nπd²
# neighbours for each spatial location. Use this information to choose k in a
# way that makes for a fair comparison between the two approaches.
small_n = 30
large_n = 300
d = ξ.r
k = ceil(Int, large_n*π*d^2)
neighbour_parameter = neighbours == "radius" ? d : k

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
θ_val,   Z_val   = variableirregularsetup(ξ, small_n, K = K_val, m = M, neighbour_parameter = neighbour_parameter)
θ_train, Z_train = variableirregularsetup(ξ, small_n, K = K_train, m = M, neighbour_parameter = neighbour_parameter)
train(gnn1, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_GNN1", epochs = epochs)

# GNN estimator trained with a fixed large n
seed!(1)
θ_val,   Z_val   = variableirregularsetup(ξ, large_n, K = K_val, m = M, neighbour_parameter = neighbour_parameter)
θ_train, Z_train = variableirregularsetup(ξ, large_n, K = K_train, m = M, neighbour_parameter = neighbour_parameter)
train(gnn2, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_GNN2", epochs = epochs)

# GNN estimator trained with a range of n
seed!(1)
θ_val,   Z_val   = variableirregularsetup(ξ, small_n:large_n, K = K_val, m = M, neighbour_parameter = neighbour_parameter)
θ_train, Z_train = variableirregularsetup(ξ, small_n:large_n, K = K_train, m = M, neighbour_parameter = neighbour_parameter)
train(gnn3, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_GNN3", epochs = epochs)

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

	println("	Running MAP estimator...")
	# Convert Z from a graph to a matrix (required for MAP)
	ξ = (ξ..., θ₀ = θ.θ) # initialise the MAP to the true parameters
	Z = broadcast(z -> reshape(z.ndata.x, :, 1), Z)
	assessment = merge(
		assessment,
		assess([MAP], θ, Z; estimator_names = ["MAP"], parameter_names = ξ.parameter_names, use_gpu = false, use_ξ = true, ξ = ξ, verbose=false)
		)

	return assessment
end

function assessestimators(n, ξ, K::Integer)
	println("	Assessing estimators with n = $n...")
	# test set for estimating the risk function
	seed!(1)
	θ, Z, ξ = variableirregularsetup(ξ, n, K = K, m = M, neighbour_parameter = neighbour_parameter, return_ξ = true)
	assessment = assessestimators(θ, Z, ξ)
	assessment.df[:, :n] .= n

	return assessment
end

assessment = [assessestimators(n, ξ, K_test) for n ∈ [30, 60, 100, 200, 300, 500, 750, 1000]]
assessment = merge(assessment...)
CSV.write(path * "/estimates_test.csv", assessment.df)
