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

path = "intermediates/$model"
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

# ------------------------------------------------------
# ---- Experiment: Comparing GNNs, CNNs, and DNNs ----
# ------------------------------------------------------

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

# GNN estimator that includes spatial distance
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

	assessments = []
	push!(assessments, assess([CNN], θ, Z; estimator_names = ["CNN"], parameter_names = ξ.parameter_names))
	push!(assessments, assess([DNN], θ, [reshapedataDNN(Z)]; estimator_names = ["DNN"], parameter_names = ξ.parameter_names))
	push!(assessments, assess([GNN, WGNN], θ, [reshapedataGNN(Z, g)]; estimator_names = ["GNN", "WGNN"], parameter_names = ξ.parameter_names))
	assessment = merge(assessments...)

	return assessment
end

# A large set of parameters for computing the risk function
seed!(1)
θ = Parameters(ξ, K_test)
Z = simulate(θ, M)
assessment = assessestimators(θ, Z, ξ, g)
CSV.write(path * "/estimates_test.csv", assessment.df)
CSV.write(path * "/runtime_test.csv", assessment.runtime)

# Focus on a small number of parameters for visualising the joint distribution
seed!(1)
θ = Parameters(ξ, 5)
Z = simulate(θ, M, 100)
assessment = assessestimators(θ, Z, ξ, g)
CSV.write(path * "/estimates_scenarios.csv", assessment.df)
CSV.write(path * "/runtime_scenarios.csv", assessment.runtime)
