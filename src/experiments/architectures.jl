# ----------------------------------------------------
# ---- Experiment: Comparing GNNs, CNNs, and DNNs ----
# ----------------------------------------------------

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

path = "intermediates/experiments/architectures/$model"
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

# The number of epochs used during training: note that early stopping means that
# we never really train for the full amount of epochs
epochs = quick ? 2 : 1000

# ---- Setup ----

A = adjacencymatrix(ξ.D, ξ.r)
g = GNNGraph(A)

seed!(1)
cnn = DeepSet(cnnarchitecture(p)...)
dnn = DeepSet(dnnarchitecture(n, p)...)
gnn = gnnarchitecture(p; propagation = "GraphConv")
wgnn = gnnarchitecture(p; propagation = "WeightedGraphConv")

# Compare the number of trainable parameters
#TODO the differing number of parameters means that overfitting is a possibility: should use on-the-fly simulation
nparams(cnn)  # 636062
nparams(dnn)  # 238658
nparams(gnn)  # 181890
nparams(wgnn) # 181894

# Sample parameters and simulate training/validation data
seed!(1)
θ_val   = Parameters(K_val, ξ, J = 5)
θ_train = Parameters(K_train, ξ, J = 5)
Z_val   = [simulate(θ_val, mᵢ)   for mᵢ ∈ m]
Z_train = [simulate(θ_train, mᵢ) for mᵢ ∈ m]


# # Testing (on my office linux)
# Z = Z_val[1][1:10];
# @time cnn(Z);  # 0.053236 seconds (359 allocations: 1.093 MiB)
# Z = Z |> gpu;
# cnn = cnn |> gpu;
# @time cnn(Z);  # 0.000693 seconds (1.78 k allocations: 101.469 KiB)
#
# Z = reshapedataGNN(Z_val[1][1:10], g);
# @time gnn(Z);  # 0.022901 seconds (2.15 k allocations: 83.138 MiB)
# @time wgnn(Z); # 0.050404 seconds (2.28 k allocations: 221.230 MiB, 6.87% gc time)
# Z = Z |> gpu;
# gnn  = gnn |> gpu;
# wgnn = wgnn |> gpu;
# @time gnn(Z);  # 0.004413 seconds (5.45 k allocations: 282.469 KiB)
# @time wgnn(Z); # 0.008089 seconds (5.77 k allocations: 298.469 KiB)


# ---- Training ----

@info "training the CNN..."
trainx(
	cnn, θ_train, θ_val, reshapedataCNN.(Z_train), reshapedataCNN.(Z_val), savepath = path * "/runs_CNN", epochs = epochs
)

@info "training the DNN..."
trainx(
	dnn, θ_train, θ_val,
	reshapedataDNN.(Z_train), reshapedataDNN.(Z_val),
	savepath = path * "/runs_DNN",
	epochs = epochs
)

@info "training the GNN..."
trainx(
  gnn, θ_train, θ_val,
  reshapedataGNN.(Z_train, Ref(g)), reshapedataGNN.(Z_val, Ref(g)),
  savepath = path * "/runs_GNN",
  epochs = epochs
)

@info "training the spatially-weighted GNN..."
trainx(
  wgnn, θ_train, θ_val,
  reshapedataGNN.(Z_train, Ref(g)), reshapedataGNN.(Z_val, Ref(g)),
  savepath = path * "/runs_WGNN",
  epochs = epochs
)


# ---- Load the trained estimators ----

Flux.loadparams!(cnn, loadbestweights(path * "/runs_CNN_m$M"))
Flux.loadparams!(dnn, loadbestweights(path * "/runs_DNN_m$M"))
Flux.loadparams!(gnn, loadbestweights(path * "/runs_GNN_m$M"))
Flux.loadparams!(wgnn,loadbestweights(path * "/runs_WGNN_m$M"))

# ---- Testing ----

function assessestimators(θ, Z, ξ, g)

	assessments = []
	push!(assessments, assess([MAP], θ, Z; estimator_names = ["MAP"], parameter_names = ξ.parameter_names, use_gpu = false, use_ξ = true, ξ = ξ))
	push!(assessments, assess([cnn], θ, reshapedataCNN(Z); estimator_names = ["CNN"], parameter_names = ξ.parameter_names))
	push!(assessments, assess([dnn], θ, reshapedataDNN(Z); estimator_names = ["DNN"], parameter_names = ξ.parameter_names))
	push!(assessments, assess([gnn, wgnn], θ, reshapedataGNN(Z, g); estimator_names = ["GNN", "WGNN"], parameter_names = ξ.parameter_names))
	assessment = merge(assessments...)

	return assessment
end

# A large set of parameters for computing the risk function
seed!(1)
θ = Parameters(K_test, ξ)
Z = simulate(θ, M)
ξ = (ξ..., θ₀ = θ.θ)
assessment = assessestimators(θ, Z, ξ, g)
CSV.write(path * "/estimates_test.csv", assessment.df)

# Focus on a small number of parameters for visualising the joint distribution
seed!(1)
K_scenarios = 5
θ = Parameters(K_scenarios, ξ)
Z = simulate(θ, M, 100)
ξ = (ξ..., θ₀ = θ.θ)
assessment = assessestimators(θ, Z, ξ, g)
CSV.write(path * "/estimates_scenarios.csv", assessment.df)

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
CSV.write(path * "/Z.csv", df)
