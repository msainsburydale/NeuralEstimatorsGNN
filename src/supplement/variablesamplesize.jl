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

path = "intermediates/supplement/variablesamplesize/$model"
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
# we never really train for the full amount of epochs
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

# GNN estimator trained with a fixed small n
seed!(1)
θ_val   = Parameters(K_val,   ξ, small_n, J = J)
θ_train = Parameters(K_train, ξ, small_n, J = J)
train(gnn1, θ_train, θ_val, simulate, m = M, savepath = path * "/runs_GNN1", epochs = epochs, epochs_per_Z_refresh = 3)

# GNN estimator trained with a fixed large n
seed!(1)
θ_val   = Parameters(K_val,   ξ, large_n, J = J)
θ_train = Parameters(K_train, ξ, large_n, J = J)
train(gnn2, θ_train, θ_val, simulate, m = M, savepath = path * "/runs_GNN2", epochs = epochs, epochs_per_Z_refresh = 3)

# GNN estimator trained with a range of n
seed!(1)
θ_val   = Parameters(K_val,   ξ, small_n:large_n, J = J)
θ_train = Parameters(K_train, ξ, small_n:large_n, J = J)
train(gnn3, θ_train, θ_val, simulate, m = M, savepath = path * "/runs_GNN3", epochs = epochs, epochs_per_Z_refresh = 3)

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

# Dummy run to compile code for more accurate timings
assessment = [assessestimators(n, ξ, K_test) for n ∈ [30, 60]]

# full experiment
assessment = [assessestimators(n, ξ, K_test) for n ∈ [30, 60, 100, 200, 350, 500, 750, 1000, 1500, 2000]]
assessment = merge(assessment...)
CSV.write(path * "/estimates.csv", assessment.df)
CSV.write(path * "/runtime.csv", assessment.runtime)


# ---- Accurately assess the run-time for a single data set ----

gnn1  = gnn1 |> gpu
gnn2  = gnn2 |> gpu
gnn3  = gnn3 |> gpu

seed!(1)
times = []
for n ∈ [32, 64, 128, 256, 512, 1024, 1500, 2048] # mostly use powers of 2

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
	t_gnn1 = @belapsed gnn1(Z)
	t_gnn2 = @belapsed gnn2(Z)
	t_gnn3 = @belapsed gnn3(Z)

	# Store the run times as a data frame
	t = DataFrame(time = [t_gnn1, t_gnn2, t_gnn3, t_ml], estimator = ["GNN1", "GNN2", "GNN3", "ML"], n = n)
	push!(times, t)
end
times = vcat(times...)
CSV.write(path * "/runtime.csv", times)
