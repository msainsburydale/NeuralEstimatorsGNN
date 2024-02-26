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

M = maximum(m)
using NeuralEstimators
using NeuralEstimatorsGNN
using GraphNeuralNetworks
using BenchmarkTools
using DataFrames
using CSV
using NamedArrays

include(joinpath(pwd(), "src/$model/model.jl"))
include(joinpath(pwd(), "src/architecture.jl"))
p = ξ.p
n = ξ.n

path = "intermediates/$model"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
J = 3
K_train = 10_000
K_val   = K_train ÷ 10
if quick
	K_train = K_train ÷ 10
	K_val   = K_val   ÷ 10
end
K_test = K_val

# ---- Estimator ----

# point estimator:
pointestimator = gnnarchitecture(p)
pointestimator_path = joinpath(path, "runs_GNN_m$M")
Flux.loadparams!(pointestimator, loadbestweights(pointestimator_path))

# Credible-interval estimator:
seed!(1)
v = gnnarchitecture(p; final_activation = identity)
Flux.loadparams!(v, loadbestweights(pointestimator_path)) # pretrain with point estimator
Ω = ξ.Ω
a = [minimum.(values(Ω))...]
b = [maximum.(values(Ω))...]
g = Compress(a, b)
intervalestimator = IntervalEstimator(v, g)

# ---- Training ----

epochs = quick ? 20 : 200

if !skip_training

	seed!(1)
	@info "Sampling parameter vectors used for validation..."
	θ_val = Parameters(K_val, ξ, n, J = J)
	@info "Sampling parameter vectors used for training..."
	θ_train = Parameters(K_train, ξ, n, J = J)
	@info "training the GNN..."
	trainx(intervalestimator, θ_train, θ_val, simulate, m, savepath = path * "/runs_GNN_CI", epochs = epochs, batchsize = 16, epochs_per_Z_refresh = 3, stopping_epochs = 3)
end

# ---- Load the trained estimator ----

Flux.loadparams!(intervalestimator, loadbestweights(path * "/runs_GNN_CI_m$M"))

# ---- Empirical coverage ----

# Simulate test data
seed!(1)
K_test = quick ? 100 : 3000
θ_test = Parameters(K_test, ξ, n, J = 1)
Z_test = simulate(θ_test, M)

# Assessment: Quantile estimator
assessment = assess(intervalestimator, θ_test, Z_test, estimator_name = "quantile", parameter_names = ξ.parameter_names)

# Assessment: Bootstrap
B = quick ? 50 : 500
θ̂_test = estimateinbatches(pointestimator, Z_test)
θ̂_test = Parameters(θ̂_test, θ_test.locations, ξ)
Z_boot = [simulate(θ̂_test, M) for _ ∈ 1:B]
Z_boot = map(1:K_test) do k
	[z[k] for z ∈ Z_boot]
end
assessment_boot = assess(pointestimator, θ_test, Z_test, boot = Z_boot, estimator_name = "bootstrap", parameter_names = ξ.parameter_names)

# Coverage
cov1 = coverage(assessment)
cov2 = coverage(assessment_boot)
cov = vcat(cov1, cov2)
CSV.write(joinpath(path, "uq_coverage.csv"), cov)


# ---- Run-time assessment ----

# Accurately assess the run-time for a single data set
θ = Parameters(1, ξ, n; cluster_process = false) # don't use a cluster process so that we can specify n exactly, rather than simply E(n)
S = θ.locations
Z = simulate(θ, M)
Z = Z |> gpu

# Quantile estimator
intervalestimator = intervalestimator |> gpu
t1 = @belapsed intervalestimator(Z)

# Bootstrap
function bs(pointestimator, Z, S, B, ξ)
	θ̂ = pointestimator(Z)
	θ̂ = θ̂ |> cpu
	θ̂ = Parameters(θ̂, S, ξ)
	Z_boot = [simulate(θ̂, M)[1] for _ ∈ 1:B]
	estimateinbatches(pointestimator, Z_boot)
end
pointestimator = pointestimator |> gpu
t2 = @belapsed bs($pointestimator, $Z, $S, $B, $ξ)

# Save the runtime
t = DataFrame(time = [t1, t2], estimator = ["quantile", "bootstrap"])
CSV.write(joinpath(path, "uq_runtime.csv"), t)
