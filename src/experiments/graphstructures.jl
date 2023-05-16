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
M = maximum(m)

using NeuralEstimators
using NeuralEstimatorsGNN
using GraphNeuralNetworks
using CSV

include(joinpath(pwd(), "src/$model/model.jl"))
include(joinpath(pwd(), "src/architecture.jl"))

path = "intermediates/experiments/graphstructures/$model"
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


# ---- Estimators ----

seed!(1)
gnn  = gnnarchitecture(p)
wgnn = gnnarchitecture(p; propagation = "WeightedGraphConv")
gnn_Svariable  = gnnarchitecture(p)
wgnn_Svariable = gnnarchitecture(p; propagation = "WeightedGraphConv")


# ---- Training ----

# Construct a specific set of irregular locations, S
seed!(1)
S = rand(n, 2)
D = pairwise(Euclidean(), S, S, dims = 1)
A = adjacencymatrix(D, ϵ)
g = GNNGraph(A)
ξ = (ξ..., D = D) # update ξ to contain the new distance matrix D
θ_val,   Z_val   = irregularsetup(ξ, g, K = K_val, m = M)
θ_train, Z_train = irregularsetup(ξ, g, K = K_train, m = M)

# GNN estimators trained under S
seed!(1)
train(gnn, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_GNN_S", epochs = epochs)
seed!(1)
train(wgnn, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_WGNN_S", epochs = epochs)

# GNN estimators trained under a variable set of irregular locations {Sₖ : k = 1, …, K}
θ_val,   Z_val   = variableirregularsetup(ξ, K = K_val, m = M, n = n, ϵ = ϵ)
θ_train, Z_train = variableirregularsetup(ξ, K = K_train, m = M, n = n, ϵ = ϵ)
seed!(1)
train(gnn_Svariable, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_GNN_Svariable", epochs = epochs)
seed!(1)
train(wgnn_Svariable, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_WGNN_Svariable", epochs = epochs)

# ---- Load the trained estimators ----

Flux.loadparams!(gnn,  loadbestweights(path * "/runs_GNN_S"))
Flux.loadparams!(wgnn, loadbestweights(path * "/runs_WGNN_S"))
Flux.loadparams!(gnn_Svariable,  loadbestweights(path * "/runs_GNN_Svariable"))
Flux.loadparams!(wgnn_Svariable, loadbestweights(path * "/runs_WGNN_Svariable"))

# ---- Assess the estimators ----

function assessestimators(θ, Z, g, ξ)
	assess(
		[gnn, gnn_Svariable, wgnn, wgnn_Svariable],
		θ,
		reshapedataGNN(Z, g);
		estimator_names = ["GNN_S", "GNN_Svariable", "WGNN_S", "WGNN_Svariable"],
		parameter_names = ξ.parameter_names
	)
end

function assessestimators(S, ξ, K::Integer, set::String)

	D = pairwise(Euclidean(), S, S, dims = 1)
	A = adjacencymatrix(D, ϵ)
	g = GNNGraph(A)
	ξ = (ξ..., D = D) # update ξ to contain the new distance matrix D

	# test set for estimating the risk function
	θ = Parameters(K_test, ξ)
	Z = simulate(θ, M)
	assessment = assessestimators(θ, Z, g, ξ)
	CSV.write(path * "/estimates_test_$set.csv", assessment.df)

	# small number of parameters for visualising the sampling distributions
	θ = Parameters(5, ξ)
	Z = simulate(θ, M, 100)
	assessment = assessestimators(θ, Z, g, ξ)
	CSV.write(path * "/estimates_scenarios_$set.csv", assessment.df)

	return 0
end

# Test with respect to the fixed set of irregular uniformly sampled locations, S
seed!(1)
assessestimators(S, ξ, K_test, "S")

# Test with respect to another specific set of irregular uniformly sampled locations, S̃
seed!(2)
S̃ = rand(n, 2)
assessestimators(S̃, ξ, K_test, "Stilde")

# Test whether a GNN trained on uniformly sampled locations does well on
# clustered data locations. The pattern looks like this:
#         . . .
#         . . .
#         . . .
#  . . .
#  . . .
#  . . .
seed!(0)
S₁ = 0.5 * rand(n÷2, 2)
S₂ = 0.5 * rand(n÷2, 2) .+ 0.5
Sclustered = vcat(S₁, S₂)
assessestimators(Sclustered, ξ, K_test, "Sclustered")
