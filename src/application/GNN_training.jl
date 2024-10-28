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
using CUDA

include(joinpath(pwd(), "src/$model/model.jl"))
include(joinpath(pwd(), "src/architecture.jl"))
p = ξ.p

path = "intermediates/application"
if !isdir(path) mkpath(path) end

# Maximum number of epochs used during training
epochs = quick ? 2 : 200


# ---- Training data ----

K = quick ? 1000 : 10000
n = vcat(repeat(100:1000, 20), 1001:2000)

seed!(1)
J = 3
@info "Sampling parameter vectors used for validation..."
θ_val = Parameters(K ÷ 10, ξ, n, J = J)
@info "Sampling parameter vectors used for training..."
θ_train = Parameters(K, ξ, n, J = J)

# ---- Point estimator ----

seed!(1)
pointestimator = gnnarchitecture(p)
@info "training the GNN-based point estimator..."
train(pointestimator, θ_train, θ_val, simulate, m = 1, savepath = joinpath(path, "pointestimator"), epochs = epochs, batchsize = 16, epochs_per_Z_refresh = 3)

# ---- Marginal posterior quantile estimator ----

v = gnnarchitecture(p; final_activation = identity)

# pretrain with point estimator
loadpath  = joinpath(path, "pointestimator", "best_network.bson")
@load loadpath model_state
Flux.loadmodel!(v, model_state)

Ω = ξ.Ω
a = [minimum.(values(Ω))...]
b = [maximum.(values(Ω))...]
g = Compress(a, b)
intervalestimator = IntervalEstimator(v, g)

@info "training the GNN-based quantile estimator..."
train(intervalestimator, θ_train, θ_val, simulate, m = 1, savepath = path * "/intervalestimator", epochs = epochs, batchsize = 16, epochs_per_Z_refresh = 3)
