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
using DataFrames
using GraphNeuralNetworks
using CSV

include(joinpath(pwd(), "src/$model/model.jl"))
include(joinpath(pwd(), "src/$model/MAP.jl"))
include(joinpath(pwd(), "src/architecture.jl"))

path = "intermediates/supplement/graphstructures/$model"
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
n = ξ.n
neighbour_parameter = ξ.r

# The number of epochs used during training: note that early stopping means that
# we never really train for the full amount of epochs
epochs = quick ? 2 : 1000

# ---- Estimators ----

seed!(1)
gnn = gnnarchitecture(p)
gnn_Svariable = deepcopy(gnn)
gnn_Sclustered = deepcopy(gnn)
gnn_Smatern = deepcopy(gnn)

# ---- Training ----

# Estimator trained under a specific set of irregular locations, S
seed!(1)
S = rand(n, 2)
D = pairwise(Euclidean(), S, S, dims = 1)
A = adjacencymatrix(D, neighbour_parameter)
g = GNNGraph(A)
ξ = (ξ..., D = D) # update ξ to contain the new distance matrix D
θ_val,   Z_val   = irregularsetup(ξ, g, K = K_val, m = M)
θ_train, Z_train = irregularsetup(ξ, g, K = K_train, m = M)
seed!(1)
train(gnn, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_GNN_S", epochs = epochs)

# Estimator trained under a variable set of irregular locations {Sₖ : k = 1, …, K}
θ_val,   Z_val   = variableirregularsetup(ξ, n, K = K_val, m = M, neighbour_parameter = neighbour_parameter)
θ_train, Z_train = variableirregularsetup(ξ, n, K = K_train, m = M, neighbour_parameter = neighbour_parameter)
seed!(1)
train(gnn_Svariable, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_GNN_Svariable", epochs = epochs)

# Estimator trained under a specific set of clustered locations. The pattern
# looks like this:
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
D = pairwise(Euclidean(), Sclustered, Sclustered, dims = 1)
A = adjacencymatrix(D, neighbour_parameter)
g = GNNGraph(A)
ξ = (ξ..., D = D) # update ξ to contain the new distance matrix D
θ_val,   Z_val   = irregularsetup(ξ, g, K = K_val, m = M)
θ_train, Z_train = irregularsetup(ξ, g, K = K_train, m = M)
seed!(1)
train(gnn_Sclustered, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_GNN_Sclustered", epochs = epochs)


# Estimator trained under a Matern clustering process
seed!(1)
θ_val,   Z_val   = variableirregularsetup(ξ, n, K = K_val, m = M, neighbour_parameter = neighbour_parameter, clustering = true)
θ_train, Z_train = variableirregularsetup(ξ, n, K = K_train, m = M, neighbour_parameter = neighbour_parameter, clustering = true)
seed!(1)
train(gnn_Smatern, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_GNN_Smatern", epochs = epochs)


# ---- Load the trained estimators ----

Flux.loadparams!(gnn, loadbestweights(path * "/runs_GNN_S"))
Flux.loadparams!(gnn_Svariable, loadbestweights(path * "/runs_GNN_Svariable"))
Flux.loadparams!(gnn_Sclustered, loadbestweights(path * "/runs_GNN_Sclustered"))
Flux.loadparams!(gnn_Smatern, loadbestweights(path * "/runs_GNN_Smatern"))


# ---- Assess the estimators ----

function assessestimators(θ, Z, g, ξ)
	assessment = assess(
		[gnn, gnn_Svariable, gnn_Sclustered, gnn_Smatern],
		θ, reshapedataGNN(Z, g);
		estimator_names = ["GNN_S", "GNN_Svariable", "GNN_Sclustered", "GNN_Smatern"],
		parameter_names = ξ.parameter_names
	)
	assessment = merge(assessment, assess([MAP], θ, Z; estimator_names = ["MAP"], parameter_names = ξ.parameter_names, use_gpu = false, use_ξ = true, ξ = ξ))
	return assessment
end

function assessestimators(S, ξ, K::Integer, set::String)

	D = pairwise(Euclidean(), S, S, dims = 1)
	A = adjacencymatrix(D, neighbour_parameter)
	g = GNNGraph(A)
	ξ = (ξ..., D = D) # update ξ to contain the new distance matrix D (needed for simulation and MAP estimation)

	# test set for estimating the risk function
	seed!(1)
	θ = Parameters(K_test, ξ)
	Z = simulate(θ, M)
	ξ = (ξ..., θ₀ = θ.θ)
	assessment = assessestimators(θ, Z, g, ξ)
	CSV.write(path * "/estimates_test_$set.csv", assessment.df)

	# small number of parameters for visualising the sampling distributions
	K_scenarios = 5
	seed!(1)
	θ = Parameters(K_scenarios, ξ)
	Z = simulate(θ, M, 100)
	ξ = (ξ..., θ₀ = θ.θ)
	assessment = assessestimators(θ, Z, g, ξ)
	CSV.write(path * "/estimates_scenarios_$set.csv", assessment.df)

	# save spatial fields for plotting
	Z = Z[1:K_scenarios] # only need one field per parameter configuration
	colons  = ntuple(_ -> (:), ndims(Z[1]) - 1)
	z  = broadcast(z -> vec(z[colons..., 1]), Z) # save only the first replicate of each parameter configuration
	z  = vcat(z...)
	d  = prod(size(Z[1])[1:end-1])
	k  = repeat(1:K_scenarios, inner = d)
	s1 = repeat(S[:, 1], K_scenarios)
	s2 = repeat(S[:, 2], K_scenarios)
	df = DataFrame(Z = z, k = k, s1 = s1, s2 = s2)
	CSV.write(path * "/Z_$set.csv", df)

	return 0
end

# Test with respect to the fixed set of irregular uniformly sampled locations, S
seed!(1)
S = rand(n, 2)
assessestimators(S, ξ, K_test, "S")

# Test with respect to another specific set of irregular uniformly sampled locations, S̃
seed!(2)
S̃ = rand(n, 2)
assessestimators(S̃, ξ, K_test, "Stilde")

# Test whether a GNN trained on uniformly sampled locations does well on
# clustered data locations.
assessestimators(Sclustered, ξ, K_test, "Sclustered")
