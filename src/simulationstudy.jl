using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--model"
		help = "A relative path to the folder of the assumed model; this folder should contain scripts for defining the parameter configurations in Julia and for data simulation."
		arg_type = String
		required = true
	"--skip_training"
		help = "A flag controlling whether or not we should skip training the estimators: useful for running the assessments without retraining the estimators."
		action = :store_true
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true
	"--m"
		help = "The sample size to use during training. If multiple samples sizes are given as a vector, multiple neural estimators will be trained."
		arg_type = String
end
parsed_args = parse_args(arg_table)

model         = parsed_args["model"]
skip_training = parsed_args["skip_training"]
quick         = parsed_args["quick"]
m = let expr = Meta.parse(parsed_args["m"])
    @assert expr.head == :vect
    Int.(expr.args)
end

# model="GP/nuFixed"
# m=[1]
# skip_training = true
# quick=true

M = maximum(m)
using NeuralEstimators
using NeuralEstimatorsGNN
using BenchmarkTools
using CSV
using DataFrames
using Distances
using GraphNeuralNetworks

include(joinpath(pwd(), "src/$model/model.jl"))
if model != "SPDE" include(joinpath(pwd(), "src/$model/ML.jl")) end
include(joinpath(pwd(), "src/architecture.jl"))

path = "intermediates/$model"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
K_train = 10_000
K_val   = K_train ÷ 10
if quick
	K_train = K_train ÷ 10
	K_val   = K_val   ÷ 10
end
K_test = K_val

p = ξ.p
n = ξ.n

# The number of epochs used during training: note that early stopping means that
# we never don't usually train for the full amount of epochs
epochs = quick ? 20 : 200

# ---- Estimators ----

seed!(1)
gnn = gnnarchitecture(p)

# ---- Training ----

J = 3

if !skip_training

	seed!(1)
	@info "Sampling parameter vectors used for validation..."
	θ_val = Parameters(K_val, ξ, n, J = J)
	@info "Sampling parameter vectors used for training..."
	θ_train = Parameters(K_train, ξ, n, J = J)
	@info "training the GNN..."
	trainx(gnn, θ_train, θ_val, simulate, m, savepath = path * "/runs_GNN", epochs = epochs, batchsize = 16, epochs_per_Z_refresh = 3, stopping_epochs = 3)

end

# ---- Load the trained estimator ----

Flux.loadparams!(gnn,  loadbestweights(path * "/runs_GNN_m$M"))

# ---- Run-time assessment ----

# Accurately assess the run-time for a single data set
if isdefined(Main, :ML)

	# Simulate data
	seed!(1)
	S = rand(n, 2)
	D = pairwise(Euclidean(), S, S, dims = 1)
	ξ = (ξ..., S = S, D = D) # update ξ to contain the new distance matrix D (needed for simulation and ML estimation)
	θ = Parameters(1, ξ)
	Z = simulate(θ, M; convert_to_graph = false)

	# ML estimates (initialised to the prior mean)
	θ₀ = mean.([ξ.Ω...])
	ξ = (ξ..., θ₀ = θ₀)
	t_ml = @belapsed ML(Z, ξ)

	# GNN estimates
	g = θ.graphs[1]
	Z = reshapedataGNN(Z, g)
	Z = Z |> gpu
	gnn  = gnn|> gpu
	t_gnn = @belapsed gnn(Z)

	# Save the runtime
	t = DataFrame(time = [t_gnn, t_ml], estimator = ["GNN", "ML"])
	CSV.write(path * "/runtime.csv", t)

end

# ---- Assess the estimators ----

function assessestimators(θ, Z, ξ)

	# Convert the data to a graph
	g = θ.graphs[1]
	Z_graph = reshapedataGNN(Z, g)

	# Assess the GNN
	assessment = assess(
		[gnn], θ, Z_graph;
		estimator_names = ["GNN"],
		parameter_names = ξ.parameter_names
	)

	# Assess the ML estimator (if it is defined)
	if isdefined(Main, :ML)
		ξ = (ξ..., θ₀ = θ.θ)
		assessment = merge(
			assessment,
			assess([ML], θ, Z; estimator_names = ["ML"], use_ξ = true, ξ = ξ)
		)
	end

	return assessment
end

function assessestimators(ξ, set::String)

	# Generate spatial locations and construct distance matrix
	S = spatialconfigurations(n, set)
	D = pairwise(Euclidean(), S, S, dims = 1)
	ξ = (ξ..., S = S, D = D) # update ξ to contain the new distance matrix D (needed for simulation and ML estimation)

	# test set for estimating the risk function
	θ = Parameters(K_test, ξ)
	Z = simulate(θ, M, convert_to_graph = false)
	assessment = assessestimators(θ, Z, ξ)
	CSV.write(path * "/estimates_test_$set.csv", assessment.df)
	CSV.write(path * "/runtime_test_$set.csv", assessment.runtime)

	# small number of parameters for visualising the sampling distributions
	K_scenarios = 5
	θ = Parameters(K_scenarios, ξ)
	J = quick ? 10 : 100
	Z = simulate(θ, M, J, convert_to_graph = false)
	assessment = assessestimators(θ, Z, ξ)
	CSV.write(path * "/estimates_scenarios_$set.csv", assessment.df)
	CSV.write(path * "/runtime_scenarios_$set.csv", assessment.runtime)

	# save spatial fields for plotting
	Z = Z[1:K_scenarios] # only need one field per parameter configuration
	colons  = ntuple(_ -> (:), ndims(Z[1]) - 1)
	z  = broadcast(z -> vec(z[colons..., 1]), Z) # save only the first replicate of each parameter configuration
	z  = vcat(z...)
	z  = broadcast(ξ.invtransform, z)
	d  = prod(size(Z[1])[1:end-1])
	k  = repeat(1:K_scenarios, inner = d)
	s1 = repeat(S[:, 1], K_scenarios)
	s2 = repeat(S[:, 2], K_scenarios)
	df = DataFrame(Z = z, k = k, s1 = s1, s2 = s2)
	CSV.write(path * "/Z_$set.csv", df)

	return 0
end


# Test with respect to a set of uniformly sampled locations
#  .   . . . .
#  . . . . .
#  . . . . . .
#  .   . .   .
#  . . . . . .
#  .   . . .
seed!(1)
assessestimators(ξ, "uniform")

# Test with respect to locations sampled only in the first and third quadrants
#         . . .
#         . . .
#         . . .
#  . . .
#  . . .
#  . . .
seed!(1)
assessestimators(ξ, "quadrants")


# Test with respect to locations with mixed sparsity.
# Divide the domain into 9 cells: have the
# central cell being very densely populated; the corner cells sparse; and the
# side cells empty.
# . .         . .
#   . .       .   .
#               .
#       . . .
#       . . .
#       . . .
# . .         .
#    .        . . .
# .             .
seed!(1)
assessestimators(ξ, "mixedsparsity")


# Test with respect to locations with a cup shape ∪.
# . .           . .
# . .           . .
# . .           . .
# . .           . .
# . .           . .
# . .           . .
# . . . . . . . . .
# . . . . . . . . .
# . . . . . . . . .
#Construct by considering the domain split into three vertical strips
seed!(1)
assessestimators(ξ, "cup")
