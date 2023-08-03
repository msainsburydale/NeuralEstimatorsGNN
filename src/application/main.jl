using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true
end
parsed_args = parse_args(arg_table)

quick = parsed_args["quick"]
model="GP/nuFixed"
m=[1]
quick=true


# use a slightly longer range parameter here
Ω = (
	τ = Uniform(0.1, 1.0),
	ρ = Uniform(0.05, 0.6)
)
ξ = (ξ..., Ω = Ω)


M = maximum(m)
using NeuralEstimators
using NeuralEstimatorsGNN
using GraphNeuralNetworks
using BenchmarkTools
using DataFrames
using CSV

include(joinpath(pwd(), "src/$model/model.jl"))
if model != "SPDE" include(joinpath(pwd(), "src/$model/MAP.jl")) end
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
n = 30:750
r = ξ.r

# The number of epochs used during training: note that early stopping means that
# we never really train for the full amount of epochs
epochs = quick ? 2 : 1000


# ---- Training data ----

seed!(1)
@info "simulating training data..."
J = 3
θ_val,   Z_val   = variableirregularsetup(ξ, n, K = K_val, m = m, neighbour_parameter = r, J = J, clustering = true)
θ_train, Z_train = variableirregularsetup(ξ, n, K = K_train, m = m, neighbour_parameter = r, J = J, clustering = true)
Z_val   = Z_val[1]
Z_train = Z_train[1]


# ---- Point estimator ----

seed!(1)
gnn = gnnarchitecture(p; propagation = "WeightedGraphConv")
@info "training the point estimator..."
train(gnn, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_pointestimator", epochs = epochs, batchsize = 16)

# ---- Credible-interval estimator ----

seed!(1)
Flux.loadparams!(gnn, loadbestweights(path * "/runs_pointestimator")) # pretrain with point estimator
lower = deepcopy(gnn)
upper = deepcopy(gnn)
intervalestimator = IntervalEstimator(deepcopy(lower), deepcopy(upper))

α = 0.05f0
q = [α/2, 1-α/2] # quantiles
qloss = (θ̂, θ) -> quantileloss(θ̂, θ, gpu(q))

train(intervalestimator, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_CIestimator", epochs = epochs, batchsize = 16, loss = qloss)
