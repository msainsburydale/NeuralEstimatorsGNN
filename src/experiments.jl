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

# model="GaussianProcess/nuFixed"
# quick=true
# m=[1]
# M = maximum(m)

using NeuralEstimators
using NeuralEstimatorsGNN
using GraphNeuralNetworks
using CSV

include(joinpath(pwd(), "src/$model/Parameters.jl"))
include(joinpath(pwd(), "src/$model/Simulation.jl"))
include(joinpath(pwd(), "src/Architecture.jl"))

path = "intermediates/$model"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
K_train = 10_000
K_val   = K_train ÷ 5
K_test  = K_val
if quick
	K_train = K_train ÷ 100
	K_val   = K_val ÷ 100
	K_test  = K_test ÷ 100
end

p = ξ.p
n = size(ξ.D, 1)
ϵ = 2.0f0 # distance cutoff used to define the local neighbourhood of each node


# ----------------------
# ---- Experiment 1 ----
# ----------------------

# Use gridded data to compare estimators that use:
# i)   GNNs
# ii)  conventional CNNs
# iii) fully-connected dense neural networks (DNNs)

A = adjacencymatrix(ξ.D, ϵ = ϵ)
g = GNNGraph(A)

# CNN estimator
seed!(1)
CNN = DeepSet(cnnarchitecture(p)...)

# DNN estimator
seed!(1)
DNN = DeepSet(dnnarchitecture(n, p)...)

# GNN estimator
seed!(1)
GNN = gnnarchitecture(p)

# GNN estimator that accounts for spatial distance
seed!(1)
WGNN = gnnarchitecture(p; propagation = "WeightedGraphConv")

# Compare the number of trainable parameters
nparams(CNN)  # 636563
nparams(DNN)  # 238723
nparams(GNN)  # 182019
nparams(WGNN) # 182023

# Sample parameters and simulate training/validation data
seed!(1)
θ_val   = Parameters(ξ, K_val,   J = 10)
θ_train = Parameters(ξ, K_train, J = 10)
Z_val   = [simulate(θ_val, mᵢ)   for mᵢ ∈ m]
Z_train = [simulate(θ_train, mᵢ) for mᵢ ∈ m]

# # Testing:
# Z = reshapedataGNN(Z_val[1][1:10], g)
# @time GNN(Z)
# @time WGNN(Z)
# Z = Z |> gpu
# GNN  = GNN |> gpu
# WGNN = WGNN |> gpu
# @time GNN(Z)
# @time WGNN(Z)

# TODO GNN and WGNN currently do not scale well to larger sample sizes


# ---- Training ----

@info "training the CNN-based estimator"
train(
  CNN, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_CNN"
)

@info "training the DNN-based estimator"
train(
	DNN, θ_train, θ_val,
	reshapedataDNN.(Z_train), reshapedataDNN.(Z_val),
	savepath = path * "/runs_DNN"
)

@info "training the GNN-based estimator"
train(
  GNN, θ_train, θ_val,
  reshapedataGNN.(Z_train, Ref(g)), reshapedataGNN.(Z_val, Ref(g)),
  savepath = path * "/runs_GNN"
)

@info "training the WGNN-based estimator"
train(
  WGNN, θ_train, θ_val,
  reshapedataGNN.(Z_train, Ref(g)), reshapedataGNN.(Z_val, Ref(g)),
  savepath = path * "/runs_WGNN"
)


# ---- Load the trained estimators ----

Flux.loadparams!(CNN, loadbestweights(path * "/runs_CNN_m$M"))
Flux.loadparams!(DNN, loadbestweights(path * "/runs_DNN_m$M"))
Flux.loadparams!(GNN, loadbestweights(path * "/runs_GNN_m$M"))
Flux.loadparams!(WGNN,loadbestweights(path * "/runs_WGNN_m$M"))


# ---- Testing ----

function assessestimators(θ, Z, ξ, g)

	pnames = ξ.parameter_names

	assessments = []

	push!(
		assessments,
		assess([CNN], θ, [Z]; estimator_names = ["CNN"], parameter_names = pnames)
	)

	push!(
		assessments,
		assess([DNN], θ, [reshapedataDNN(Z)]; estimator_names = ["DNN"], parameter_names = pnames)
	)

	push!(
		assessments,
		assess([GNN, WGNN], θ, [reshapedataGNN(Z, g)]; estimator_names = ["GNN", "WGNN"], parameter_names = pnames)
	)

	assessment = merge(assessments...)

	return assessment
end

# Compute the risk function many times for accurate results
seed!(1)
assessments = map(1:10) do i
	θ = Parameters(ξ, K_test)
	Z = simulate(θ, M)
	assessment = assessestimators(θ, Z, ξ, g)
	assessment.θandθ̂[:, :trial] .= i
	assessment.runtime[:, :trial] .= i
	assessment
end
assessment = merge(assessments...)
CSV.write(path * "/estimates_test.csv", assessment.θandθ̂)
CSV.write(path * "/runtime_test.csv", assessment.runtime)

# Focus on a small number of parameters for visualising the joint distribution
seed!(1)
θ = Parameters(ξ, 5)
Z = simulate(θ, M, 100)
assessment = assessestimators(θ, Z, ξ, g)
CSV.write(path * "/estimates_scenarios.csv", assessment.θandθ̂)
CSV.write(path * "/runtime_scenarios.csv", assessment.runtime)




# ----------------------
# ---- Experiment 2 ----
# ----------------------

# Applying GNNs to different graph structures.
# Here we compare a GNN trained under a specific set of irregular locations S,
# a GNN estimator trained under a fixed set of gridded locations S̃,
# and a third trained with a variable set of irregular locations {Sₖ : k = 1, …, K},
# where K is the number of unique parameter vectors in the training set.
# We assess the estimators with respect to the set of specific irregular locations, S.
# The purpose of this experiment is to determine whether optimal inference
# requires the neural estimator to be trained specifically under the spatial
# locations of the given data set, or if a general estimator can be used that
# is close to optimal irrespective of the configuration of the spatial locations.


# ---- Training ----

function irregularsetup(ξ, g; K, m, J = 10)

	θ = Parameters(ξ, K, J = J)
	Z = [simulate(θ, mᵢ) for mᵢ ∈ m]
	Z = reshapedataGNN.(Z, Ref(g))

	return θ, Z
end

function variableirregularsetup(ξ; K, m, J = 10)

	D = map(1:K) do k
		S = 16 * rand(n, 2)
		D = pairwise(Euclidean(), S, S, dims = 1)
		D
	end
	A = adjacencymatrix.(D, ϵ = ϵ)
	g = GNNGraph.(A)

	ξ = (ξ..., D = D) # update ξ to contain the new distance matrix D
	θ = Parameters(ξ, K, J = J)
	Z = [simulate(θ, mᵢ) for mᵢ ∈ m]

	g = repeat(g, inner = J)
	Z = reshapedataGNN.(Z, Ref(g))

	return θ, Z
end

# Construct the specific set of irregular locations, S
S = 16 * rand(n, 2)
D = pairwise(Euclidean(), S, S, dims = 1)
A = adjacencymatrix(D, ϵ = ϵ)
g = GNNGraph(A)
ξ = (ξ..., D = D) # update ξ to contain the new distance matrix D
θ_val2,   Z_val2    = irregularsetup(ξ, g, K = K_val, m = M)
θ_train2, Z_train2  = irregularsetup(ξ, g, K = K_train, m = M)

# GNN estimator trained under S
seed!(1)
GNN2 = gnnarchitecture(p)
train(GNN2, θ_train2, θ_val2, Z_train2, Z_val2, savepath = path * "/runs_GNN_S")

# WGNN estimator trained under S
seed!(1)
WGNN2 = gnnarchitecture(p; propagation = "WeightedGraphConv")
train(WGNN2, θ_train2, θ_val2, Z_train2, Z_val2, savepath = path * "/runs_WGNN_S")

# Construct a variable set of irregular locations {Sₖ : k = 1, …, K}
θ_val3,   Z_val3    = variableirregularsetup(ξ, K = K_val, m = M)
θ_train3, Z_train3  = variableirregularsetup(ξ, K = K_train, m = M)

# GNN estimator trained under {Sₖ : k = 1, …, K}
seed!(1)
GNN3 = gnnarchitecture(p)
train(GNN3, θ_train3, θ_val3, Z_train3, Z_val3, savepath = path * "/runs_GNN_Svariable")

# WGNN estimator trained under {Sₖ : k = 1, …, K}
seed!(1)
WGNN3 = gnnarchitecture(p; propagation = "WeightedGraphConv")
train(WGNN3, θ_train3, θ_val3, Z_train3, Z_val3, savepath = path * "/runs_WGNN_Svariable")

# ---- Load the trained estimators ----

Flux.loadparams!(GNN,   loadbestweights(path * "/runs_GNN_m$M"))
Flux.loadparams!(GNN2,  loadbestweights(path * "/runs_GNN_S_m$M"))
Flux.loadparams!(GNN3,  loadbestweights(path * "/runs_GNN_Svariable_m$M"))
Flux.loadparams!(WGNN,  loadbestweights(path * "/runs_WGNN_m$M"))
Flux.loadparams!(WGNN2, loadbestweights(path * "/runs_WGNN_S_m$M"))
Flux.loadparams!(WGNN3, loadbestweights(path * "/runs_WGNN_Svariable_m$M"))


# ---- Testing ----

# Testing is done with respect to the fixed set of irregular locations

function assessestimators2(θ, Z, ξ, g)

	Z = reshapedataGNN(Z, g)

	assessment = assess(
		[GNN, GNN2, GNN3, WGNN, WGNN2, WGNN3], θ, [Z];
		estimator_names = ["GNN", "GNN_S", "GNN_Svariable", "WGNN", "WGNN_S", "WGNN_Svariable"],
		parameter_names = ξ.parameter_names
	)

	return assessment
end

# Sample a large set of parameters for computing the risk function
seed!(1)
assessments = map(1:10) do i
	θ = Parameters(ξ, K_test)
	Z = simulate(θ, M)
	assessment = assessestimators2(θ, Z, ξ, g)
	assessment.θandθ̂[:, :trial] .= i
	assessment.runtime[:, :trial] .= i
	assessment
end
assessment = merge(assessments...)
CSV.write(path * "/estimates_test_S.csv", assessment.θandθ̂)
CSV.write(path * "/runtime_test_S.csv", assessment.runtime)

# Focus on a small number of parameters for visualising the joint distribution
seed!(1)
θ = Parameters(ξ, 5)
Z = simulate(θ, M, 100)
assessment = assessestimators2(θ, Z, ξ, g)
CSV.write(path * "/estimates_scenarios_S.csv", assessment.θandθ̂)
CSV.write(path * "/runtime_scenarios_S.csv", assessment.runtime)
