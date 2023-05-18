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

# model="Schlather"
# m=[1, 30]
# quick=true


M = maximum(m)
using NeuralEstimators
using NeuralEstimatorsGNN
using DataFrames
using GraphNeuralNetworks
using CSV

include(joinpath(pwd(), "src/$model/model.jl"))
# include(joinpath(pwd(), "src/$model/MAP.jl")) # TODO add Pairwise likelihood
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
ϵ = ξ.ϵ

# The number of epochs used during training: note that early stopping means that
# we never really train for the full amount of epochs
epochs = quick ? 2 : 1000

# ---- Estimators ----

seed!(1)
cnn = DeepSet(cnnarchitecture(p)...)
gnn = gnnarchitecture(p)


# ---- Training ----

# Note that we have to use different training data for the two estimators,
# because CNNs require gridded data whilst GNNs require training with irregular
# data if they are to generalise well.

# CNN estimator
@info "training the CNN..."
seed!(1)
θ_val   = Parameters(K_val, ξ, J = 5)
θ_train = Parameters(K_train, ξ, J = 5)
Z_val   = simulate(θ_val, M)
Z_train = simulate(θ_train, M)
z =  reshapedataCNN(Z_val)
trainx(cnn, θ_train, θ_val, reshapedataCNN(Z_train), reshapedataCNN(Z_val), savepath = path * "/runs_CNN", epochs = epochs)

# GNN estimator
@info "training the GNN..."
@assert length(m) == 1 "Need some minor adjustments to accommodate length(m) > 1"
seed!(1)
θ_val,   Z_val   = variableirregularsetup(ξ, n, K = K_val, m = M, ϵ = ϵ)
θ_train, Z_train = variableirregularsetup(ξ, n, K = K_train, m = M, ϵ = ϵ)
train(gnn, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_GNN", epochs = epochs)


# ---- Load the trained estimators ----

Flux.loadparams!(cnn,  loadbestweights(path * "/runs_CNN_m$M"))
Flux.loadparams!(gnn,  loadbestweights(path * "/runs_GNN"))

# ---- Assess the estimators ----

#TODO

# Assess the estimators with respect to:
# - Gridded data (for comparison with the prior art, the CNN)
# - Randomly sampled irregular data
# - Clustered data

# Note that we keep n constant to facilitate comparison between data sets

# function assessestimators(θ, Z, ξ)
# 	assessment = assess(
# 		[gnn, gnn2, gnn3], θ, Z;
# 		estimator_names = ["GNN", "GNN2", "GNN3"],
# 		parameter_names = ξ.parameter_names,
# 		verbose = false
# 	)
# 	# ξ = (ξ..., θ₀ = θ.θ)
# 	# assessment = merge(assessment, assess([MAP], θ, Z; estimator_names = ["MAP"], parameter_names = ξ.parameter_names, use_gpu = false, use_ξ = true, ξ = ξ))
# 	return assessment
# end
#
# function assessestimators(n, ξ, K::Integer)
# 	println("	Assessing estimators with n = $n...")
# 	# test set for estimating the risk function
# 	seed!(1)
# 	θ, Z = variableirregularsetup(ξ, n, K = K, m = M, ϵ = ϵ)
# 	assessment = assessestimators(θ, Z, ξ)
# 	assessment.df[:, :n] .= n
#
# 	return assessment
# end
#
# assessment = [assessestimators(n, ξ, K_test) for n ∈ [30, 60, 100, 200, 300, 500, 750, 1000]]
# assessment = merge(assessment...)
# CSV.write(path * "/estimates_test.csv", assessment.df)
#
#
# # ---- Assess the estimators ----
#
# function assessestimators(θ, Z, g, ξ)
# 	assessment = assess(
# 		[gnn, gnn_Svariable, wgnn, wgnn_Svariable], θ, reshapedataGNN(Z, g);
# 		estimator_names = ["GNN_S", "GNN_Svariable", "WGNN_S", "WGNN_Svariable"],
# 		parameter_names = ξ.parameter_names
# 	)
# 	assessment = merge(assessment, assess([MAP], θ, Z; estimator_names = ["MAP"], parameter_names = ξ.parameter_names, use_gpu = false, use_ξ = true, ξ = ξ))
# 	return assessment
# end
#
# function assessestimators(S, ξ, K::Integer, set::String)
#
# 	D = pairwise(Euclidean(), S, S, dims = 1)
# 	A = adjacencymatrix(D, ϵ)
# 	g = GNNGraph(A)
# 	ξ = (ξ..., D = D) # update ξ to contain the new distance matrix D (needed for simulation and MAP estimation)
#
# 	# test set for estimating the risk function
# 	seed!(1)
# 	θ = Parameters(K_test, ξ)
# 	Z = simulate(θ, M)
# 	ξ = (ξ..., θ₀ = θ.θ)
# 	assessment = assessestimators(θ, Z, g, ξ)
# 	CSV.write(path * "/estimates_test_$set.csv", assessment.df)
#
# 	# small number of parameters for visualising the sampling distributions
# 	K_scenarios = 5
# 	seed!(1)
# 	θ = Parameters(K_scenarios, ξ)
# 	Z = simulate(θ, M, 100)
# 	ξ = (ξ..., θ₀ = θ.θ)
# 	assessment = assessestimators(θ, Z, g, ξ)
# 	CSV.write(path * "/estimates_scenarios_$set.csv", assessment.df)
#
# 	# save spatial fields for plotting
# 	Z = Z[1:K_scenarios] # only need one field per parameter configuration
# 	colons  = ntuple(_ -> (:), ndims(Z[1]) - 1)
# 	z  = broadcast(z -> vec(z[colons..., 1]), Z) # save only the first replicate of each parameter configuration
# 	z  = vcat(z...)
# 	d  = prod(size(Z[1])[1:end-1])
# 	k  = repeat(1:K_scenarios, inner = d)
# 	s1 = repeat(S[:, 1], K_scenarios)
# 	s2 = repeat(S[:, 2], K_scenarios)
# 	df = DataFrame(Z = z, k = k, s1 = s1, s2 = s2)
# 	CSV.write(path * "/Z_$set.csv", df)
#
# 	return 0
# end
#
# # Test with respect to gridded data
# pts = range(0, 1, length = 16)
# S   = expandgrid(pts, pts)
# seed!(1)
# assessestimators(S, ξ, K_test, "gridded")
#
# # Test with respect to another specific set of irregular uniformly sampled locations, S̃
# seed!(2)
# S̃ = rand(n, 2)
# assessestimators(S̃, ξ, K_test, "Stilde")
#
# # Test whether a GNN trained on uniformly sampled locations does well on
# # clustered data locations. The pattern looks like this:
# #         . . .
# #         . . .
# #         . . .
# #  . . .
# #  . . .
# #  . . .
# seed!(0)
# S₁ = 0.5 * rand(n÷2, 2)
# S₂ = 0.5 * rand(n÷2, 2) .+ 0.5
# Sclustered = vcat(S₁, S₂)
# assessestimators(Sclustered, ξ, K_test, "Sclustered")
