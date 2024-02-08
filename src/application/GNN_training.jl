using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true
end
parsed_args = parse_args(arg_table)

quick = parsed_args["quick"]
model = joinpath("GP", "nuFixed")
m=[1]

M = maximum(m)
using NeuralEstimators
using NeuralEstimatorsGNN
using GraphNeuralNetworks
using BenchmarkTools
using DataFrames
using CSV

include(joinpath(pwd(), "src/$model/model.jl"))
include(joinpath(pwd(), "src/architecture.jl"))

path = "intermediates/application/SST"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
K_train = 10_000
K_val   = K_train ÷ 10
if quick
	K_train = K_train ÷ 100
	K_val   = K_val   ÷ 100
end
K_test = K_val

p = ξ.p
n = 30:2000

# The number of epochs used during training: note that early stopping means that
# we never really train for the full amount of epochs
epochs = quick ? 2 : 200


# ---- Training data ----

seed!(1)
@info "simulating training data..."
J = 3
@info "Sampling set of parameter vectors used for validation..."
θ_val = Parameters(K_val, ξ, n, J = J)
@info "Sampling set of parameter vectors used for training..."
θ_train = Parameters(K_train, ξ, n, J = J)
@info "training the GNN..."


# ---- Point estimator ----

seed!(1)
gnn = gnnarchitecture(p)
@info "training the point estimator..."
train(gnn, θ_train, θ_val, simulate, m = 1, savepath = path * "/pointestimator", epochs = epochs, batchsize = 16, epochs_per_Z_refresh = 3)

# ---- Credible-interval estimator ----

U = gnnarchitecture(p; final_activation = identity)
V = deepcopy(U)
Flux.loadparams!(U, loadbestweights(path * "/pointestimator")) # pretrain with point estimator
Flux.loadparams!(V, loadbestweights(path * "/pointestimator")) # pretrain with point estimator
Ω = ξ.Ω
a = [minimum.(values(Ω))...]
b = [maximum.(values(Ω))...]
intervalestimator = IntervalEstimator(U, V, a, b)

α = 0.05f0
q = [α/2, 1-α/2] # quantiles
qloss = (θ̂, θ) -> quantileloss(θ̂, θ, gpu(q))

@info "training the quantile estimator..."
train(intervalestimator, θ_train, θ_val, simulate, m = 1, savepath = path * "/intervalestimator", epochs = epochs, batchsize = 16, loss = qloss, epochs_per_Z_refresh = 3)
