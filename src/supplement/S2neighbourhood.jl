# ------------------------------------------------------------------------------
#  Experiment: Definition of the node neighbourhood
#  - a disc of fixed spatial radius
#  - k-nearest neighbours for fixed k
#  - a random set of k neighbours selected within a disc of fixed spatial radius
#  - maxmin ordering (both immoral and moral versions)
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
K_test = K_val

p = ξ.p

# For uniformly sampled locations on the unit square, the probability that a
# point falls within a circle of radius r << 1 centred away from the boundary is
# simply πr². So, on average, we expect nπr² neighbours for each node. Use this
# information to choose k in a way that makes for a relatively fair comparison.
n = 250  # sample size
r = 0.15 # disc radius
# k = ceil(Int, n*π*r^2)
k = 10
k₂ = 30

# Number of epochs used during training: Early stopping means that we never train
# for the full amount of epochs
epochs = quick ? 2 : 1000


# ---- initialise the estimators ----

seed!(1)
gnn1 = gnnarchitecture(p; propagation = "WeightedGraphConv")
gnn2 = deepcopy(gnn1)
gnn2b = deepcopy(gnn1)
gnn3 = deepcopy(gnn1)
gnn3b = deepcopy(gnn1)
gnn4 = deepcopy(gnn1)
gnn5 = deepcopy(gnn1)


# ---- Training ----

cluster_process = true
epochs_per_Z_refresh = 3

# Sample parameter vectors
seed!(1)
θ_val   = Parameters(K_val,   ξ, n, J = 5, cluster_process = cluster_process)
θ_train = Parameters(K_train, ξ, n, J = 5, cluster_process = cluster_process)

@info "Training with immoral maxmin ordering with k=$k neighbours"
θ̃_val   = modifyneighbourhood(θ_val, k;   maxmin = true, moralise = false)
θ̃_train = modifyneighbourhood(θ_train, k; maxmin = true, moralise = false)
train(gnn4, θ̃_train, θ̃_val, simulate, m = m, savepath = joinpath(path, "maxmin_immoral"), epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh)

@info "Training with moral maxmin ordering with k=$k neighbours"
θ̃_val   = modifyneighbourhood(θ_val, k;   maxmin = true, moralise = true)
θ̃_train = modifyneighbourhood(θ_train, k; maxmin = true, moralise = true)
train(gnn5, θ̃_train, θ̃_val, simulate, m = m, savepath = joinpath(path, "maxmin_moral"), epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh)

@info "Training with the neighbourhood of a given node defined as a disc of fixed radius"
θ̃_val   = modifyneighbourhood(θ_val, r)
θ̃_train = modifyneighbourhood(θ_train, r)
train(gnn1, θ̃_train, θ̃_val, simulate, m = m, savepath = joinpath(path, "fixedradius"), epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh)

@info "Training with the neighbourhood of a given node defined as its k-nearest neighbours with k=$k"
θ̃_val   = modifyneighbourhood(θ_val, k)
θ̃_train = modifyneighbourhood(θ_train, k)
train(gnn2, θ̃_train, θ̃_val, simulate, m = m, savepath = joinpath(path, "knearest"), epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh)

@info "Training with the neighbourhood of a given node defined as its k-nearest neighbours with k=$(k₂)"
θ̃_val   = modifyneighbourhood(θ_val, k₂)
θ̃_train = modifyneighbourhood(θ_train, k₂)
train(gnn2b, θ̃_train, θ̃_val, simulate, m = m, savepath = joinpath(path, "knearestb"), epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh)

@info "Training with the neighbourhood of a given node a random set of k=$k neighbours selected within a disc of fixed spatial radius"
θ̃_val   = modifyneighbourhood(θ_val, r, k)
θ̃_train = modifyneighbourhood(θ_train, r, k)
train(gnn3, θ̃_train, θ̃_val, simulate, m = m, savepath = joinpath(path, "combined"), epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh)

@info "Training with the neighbourhood of a given node a random set of k=$(k₂) neighbours selected within a disc of fixed spatial radius"
θ̃_val   = modifyneighbourhood(θ_val, r, k₂)
θ̃_train = modifyneighbourhood(θ_train, r, k₂)
train(gnn3b, θ̃_train, θ̃_val, simulate, m = m, savepath = joinpath(path, "combinedb"), epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh)


# ---- Load the trained estimators ----

Flux.loadparams!(gnn1,  loadbestweights(joinpath(path, "fixedradius")))
Flux.loadparams!(gnn2,  loadbestweights(joinpath(path, "knearest")))
Flux.loadparams!(gnn2b, loadbestweights(joinpath(path, "knearestb")))
Flux.loadparams!(gnn3,  loadbestweights(joinpath(path, "combined")))
Flux.loadparams!(gnn3b, loadbestweights(joinpath(path, "combinedb")))
Flux.loadparams!(gnn4,  loadbestweights(joinpath(path, "maxmin_immoral")))
Flux.loadparams!(gnn5,  loadbestweights(joinpath(path, "maxmin_moral")))

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
	θ̃ = modifyneighbourhood(θ, k₂)
	seed!(1); Z = simulate(θ̃, m)
	assessment = merge(assessment, assess(
		[gnn2b], θ̃, Z;
		estimator_names = ["knearestb"],
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
	θ̃ = modifyneighbourhood(θ, r, k₂)
	seed!(1); Z = simulate(θ̃, m)
	assessment = merge(assessment, assess(
		[gnn3b], θ̃, Z;
		estimator_names = ["combinedb"],
		parameter_names = ξ.parameter_names,
		verbose = false
	))

	# Estimators trained with maxmin ordering
	θ̃ = modifyneighbourhood(θ, k; maxmin = true, moralise = false)
	seed!(1); Z = simulate(θ̃, m)
	assessment = merge(assessment, assess(
		[gnn4], θ̃, Z;
		estimator_names = ["maxmin_immoral"],
		parameter_names = ξ.parameter_names,
		verbose = false
	))
	θ̃ = modifyneighbourhood(θ, k; maxmin = true, moralise = true)
	seed!(1); Z = simulate(θ̃, m)
	assessment = merge(assessment, assess(
		[gnn5], θ̃, Z;
		estimator_names = ["maxmin_moral"],
		parameter_names = ξ.parameter_names,
		verbose = false
	))

	# Add sample size information
	assessment.df[:, :n] .= n

	return assessment
end

test_n = [30, 60, 100, 200, 300, 500, 750, 1000]
assessment = [assessestimators(n, ξ, K_test) for n ∈ test_n]
assessment = merge(assessment...)
CSV.write(joinpath(path, "estimates.csv"), assessment.df)
CSV.write(joinpath(path, "runtime.csv"), assessment.runtime)


# ---- Accurately assess the run-time for a single data set ----

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
  	θ̃ = modifyneighbourhood(θ, k₂)
	Z = simulate(θ̃, m)|> gpu
  	t_gnn2b = @belapsed gnn($Z)

  	# combined
    θ̃ = modifyneighbourhood(θ, r, k)
	Z = simulate(θ̃, m)|> gpu
  	t_gnn3 = @belapsed gnn($Z)
  	θ̃ = modifyneighbourhood(θ, r, k₂)
	Z = simulate(θ̃, m)|> gpu
  	t_gnn3b = @belapsed gnn($Z)

	# maxmin
	θ̃ = modifyneighbourhood(θ, k; maxmin = true, moralise = false)
	Z = simulate(θ̃, m)|> gpu
	t_gnn4 = @belapsed gnn($Z)
	θ̃ = modifyneighbourhood(θ, k; maxmin = true, moralise = true)
	Z = simulate(θ̃, m)|> gpu
	t_gnn5 = @belapsed gnn($Z)

  	# Store the run times as a data frame
  	DataFrame(time = [t_gnn1, t_gnn2, t_gnn2b, t_gnn3, t_gnn3b, t_gnn4, t_gnn5],
  	          estimator = ["fixedradius", "knearest", "knearestb", "combined", "combinedb", "maxmin_immoral", "maxmin_moral"],
  	          n = n)
  end

test_n = [32, 64, 128, 256, 512, 1024]
seed!(1)
times = []
for n ∈ test_n
	t = testruntime(n, ξ)
	push!(times, t)
end
times = vcat(times...)

CSV.write(joinpath(path, "runtime_singledataset.csv"), times)
