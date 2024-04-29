# ------------------------------------------------------------------------------
#  Experiment: Comparing definitions for the neighbourhood of a node
#  - Disc of fixed spatial radius
#  - k-nearest neighbours for fixed k
#  - Random-k neighbours within a disc of fixed spatial radius
#  - Maxmin ordering
# ------------------------------------------------------------------------------

using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true
end
parsed_args = parse_args(arg_table)
quick       = parsed_args["quick"]

model = joinpath("GP", "nuSigmaFixed")
m = 1
using NeuralEstimators
using NeuralEstimatorsGNN
using BenchmarkTools
using DataFrames
using GraphNeuralNetworks
using CSV

# Helper functions for this script: same API as adjacencymatrix()
function modifyneighbourhood(θ::Parameters, args...)

	S = θ.locations
	if !(typeof(S) <: AbstractVector) S = [S] end

	A = adjacencymatrix.(S, args...)
	graphs = GNNGraph.(A)

	Parameters(θ.θ, S, graphs, θ.chols, θ.chol_pointer, θ.loc_pointer)
end
function modifyneighbourhood(θ::Parameters, k::Integer; kwargs...)

	S = θ.locations
	if !(typeof(S) <: AbstractVector) S = [S] end

	A = adjacencymatrix.(S, k; kwargs...)
	graphs = GNNGraph.(A)

	Parameters(θ.θ, S, graphs, θ.chols, θ.chol_pointer, θ.loc_pointer)
end


include(joinpath(pwd(), "src", model, "model.jl"))
include(joinpath(pwd(), "src", "architecture.jl"))

path = "intermediates/supplement/neighbours"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
K_train = 10_000
K_val   = K_train ÷ 2
if quick
	K_train = K_train ÷ 100
	K_val   = K_val   ÷ 100
end
K_test = quick ? 50 : 1000

n = 250  # sample size
r = 0.10 # disc radius
k = 10   # number of neighbours


# ---- initialise the estimators ----

seed!(1)
p = ξ.p # number of parameters in the statistical model
gnn1 = gnnarchitecture(p)
gnn2 = deepcopy(gnn1)
gnn3 = deepcopy(gnn1)
gnn4 = deepcopy(gnn1)

# ---- Training ----

cluster_process = true
epochs_per_Z_refresh = 3
epochs = quick ? 2 : 300
stopping_epochs = epochs   # "turn off" early stopping

# Sample parameter vectors
seed!(1)
θ_val   = Parameters(K_val,   ξ, n, J = 5, cluster_process = cluster_process)
θ_train = Parameters(K_train, ξ, n, J = 5, cluster_process = cluster_process)

@info "Training with a disc of fixed radius"
θ_val   = modifyneighbourhood(θ_val, r)
θ_train = modifyneighbourhood(θ_train, r)
train(gnn1, θ_train, θ_val, simulate, m = m, savepath = joinpath(path, "fixedradius"), epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh, stopping_epochs = stopping_epochs)

@info "Training with k-nearest neighbours with k=$k"
θ_val   = modifyneighbourhood(θ_val, k)
θ_train = modifyneighbourhood(θ_train, k)
train(gnn2, θ_train, θ_val, simulate, m = m, savepath = joinpath(path, "knearest"), epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh, stopping_epochs = stopping_epochs)

@info "Training with a random set of k=$k neighbours selected within a disc of fixed spatial radius"
θ_val   = modifyneighbourhood(θ_val, r, k)
θ_train = modifyneighbourhood(θ_train, r, k)
train(gnn3, θ_train, θ_val, simulate, m = m, savepath = joinpath(path, "combined"), epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh, stopping_epochs = stopping_epochs)

@info "Training with maxmin ordering with k=$k neighbours"
θ_val   = modifyneighbourhood(θ_val, k;   maxmin = true)
θ_train = modifyneighbourhood(θ_train, k; maxmin = true)
train(gnn4, θ_train, θ_val, simulate, m = m, savepath = joinpath(path, "maxmin"), epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh, stopping_epochs = stopping_epochs)


# ---- Load the trained estimators ----

Flux.loadparams!(gnn1,  loadbestweights(joinpath(path, "fixedradius")))
Flux.loadparams!(gnn2,  loadbestweights(joinpath(path, "knearest")))
Flux.loadparams!(gnn3,  loadbestweights(joinpath(path, "combined")))
Flux.loadparams!(gnn4,  loadbestweights(joinpath(path, "maxmin")))

# ---- Assess the estimators ----

function assessestimators(n, ξ, K::Integer)
	println("	Assessing estimators with n = $n...")

	# test parameter vectors for estimating the risk function
	seed!(1)
	θ = Parameters(K, ξ, n, cluster_process = cluster_process)

	# Estimator trained with a fixed radius
	θ̃ = modifyneighbourhood(θ, r)
	seed!(1); Z = simulate(θ̃, m)
	assessment = assess(
		[gnn1], θ̃, Z;
		estimator_names = ["fixedradius"],
		parameter_names = ξ.parameter_names,
		verbose = false
	)

	# Estimators trained with k-nearest neighbours for fixed k
	θ̃ = modifyneighbourhood(θ, k)
	seed!(1); Z = simulate(θ̃, m)
	assessment = merge(assessment, assess(
		[gnn2], θ̃, Z;
		estimator_names = ["knearest"],
		parameter_names = ξ.parameter_names,
		verbose = false
	))

	# Estimator trained with a random set of k neighbours selected within a disc of fixed spatial radius
	θ̃ = modifyneighbourhood(θ, r, k)
	seed!(1); Z = simulate(θ̃, m)
	assessment = merge(assessment, assess(
		[gnn3], θ̃, Z;
		estimator_names = ["combined"],
		parameter_names = ξ.parameter_names,
		verbose = false
	))

	# Estimators trained with maxmin ordering
	θ̃ = modifyneighbourhood(θ, k; maxmin = true)
	seed!(1); Z = simulate(θ̃, m)
	assessment = merge(assessment, assess(
		[gnn4], θ̃, Z;
		estimator_names = ["maxmin"],
		parameter_names = ξ.parameter_names,
		verbose = false
	))

	# Add sample size information
	assessment.df[:, :n] .= n

	return assessment
end

test_n = [60, 100, 200, 300, 500, 750, 1000]
assessment = [assessestimators(n, ξ, K_test) for n ∈ test_n]
assessment = merge(assessment...)
CSV.write(joinpath(path, "estimates.csv"), assessment.df)
CSV.write(joinpath(path, "runtime.csv"), assessment.runtime)


# ---- Accurately assess the run-times for a single data set ----

@info "Assessing the run-time for a single data set for each estimator and each sample size n ∈ $(test_n)..."

# just use one gnn since the architectures are exactly the same
gnn  = gnn1 |> gpu

function testruntime(n, ξ)

    # Simulate locations and data
	seed!(1)
    S = rand(n, 2)
	D = pairwise(Euclidean(), S, S, dims = 1)
  	ξ = (ξ..., S = S, D = D) # update ξ to contain the new distance matrix D (needed for simulation and ML estimation)
  	θ = Parameters(1, ξ)
  	Z = simulate(θ, m; convert_to_graph = false)

  	seed!(1)
  	θ = Parameters(1, ξ, n, cluster_process = cluster_process)

  	# Fixed radius
  	θ̃ = modifyneighbourhood(θ, r)
	Z = simulate(θ̃, m)|> gpu
  	t_gnn1 = @belapsed gnn($Z)

  	# k-nearest neighbours
    θ̃ = modifyneighbourhood(θ, k)
	Z = simulate(θ̃, m)|> gpu
  	t_gnn2 = @belapsed gnn($Z)

  	# combined
    θ̃ = modifyneighbourhood(θ, r, k)
	Z = simulate(θ̃, m)|> gpu
  	t_gnn3 = @belapsed gnn($Z)

	# maxmin
	θ̃ = modifyneighbourhood(θ, k; maxmin = true)
	Z = simulate(θ̃, m)|> gpu
	t_gnn4 = @belapsed gnn($Z)

  	# Store the run times as a data frame
  	DataFrame(
	time = [t_gnn1, t_gnn2, t_gnn3, t_gnn4],
	estimator = ["fixedradius", "knearest", "combined", "maxmin"],
	n = n)
  end

seed!(1)
times = []
for n ∈ [32, 64, 128, 256, 512, 1024, 2048, 4096]
	t = testruntime(n, ξ)
	push!(times, t)
end
times = vcat(times...)

CSV.write(joinpath(path, "runtime_singledataset.csv"), times)


# ---- Sensitivity analysis with respect to k for maxmin ordering ----


all_k = [5, 10, 15, 20, 25]

for k ∈ all_k

	@info "Training GNN with maxmin ordering and k=$k"

	savepath = joinpath(path, "maxmin_k$k")

	global θ_val   = modifyneighbourhood(θ_val, k; maxmin = true)
	global θ_train = modifyneighbourhood(θ_train, k; maxmin = true)

	seed!(1)
	gnn = gnnarchitecture(p)
	train(gnn, θ_train, θ_val, simulate, m = m, savepath = savepath,
		  epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh,
		  stopping_epochs = stopping_epochs)
end

assessments = []
for n ∈ union(test_n, [2048, 4096])
  @info "Assessing the estimators with sample size n=$n"
  seed!(1)
  θ_test   = Parameters(n > 1000 ? 1 : K_test, ξ, n, J = 1) #NB only want run-times timings for n>1000
  θ_single = Parameters(1, ξ, n, J = 1)
  for k ∈ all_k

    @info "Assessing the estimator with k=$k"

    gnn = gnnarchitecture(p)
  	loadpath = joinpath(path, "maxmin_k$k")
  	Flux.loadparams!(gnn, loadbestweights(loadpath))

  	θ_test = modifyneighbourhood(θ_test, k; maxmin = true)
  	θ_single = modifyneighbourhood(θ_single, k; maxmin = true)

  	# Assess the estimator's accuracy
  	seed!(1)
  	Z_test = simulate(θ_test, m)
  	assessment = assess(
  		[gnn], θ_test, Z_test;
  		estimator_names = ["gnn_k$k"],
  		parameter_names = ξ.parameter_names,
  		verbose = false
  	)
  	assessment.df[:, :k] .= k
  	assessment.df[:, :n] .= n

  	# Accurately assess the inference time for a single data set
    gnn = gnn |> gpu
    Z_single = simulate(θ_single, m) |> gpu
    t = @belapsed $gnn($Z_single)
    assessment.df[:, :inference_time] .= t

    push!(assessments, assessment)
  end
end
assessment = merge(assessments...)
CSV.write(joinpath(path, "k_vs_n.csv"), assessment.df)

#
