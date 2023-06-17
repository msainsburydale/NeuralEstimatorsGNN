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
M = maximum(m)

model="GaussianProcess/nuFixed"
quick=true
m=[1]
M = maximum(m)

using NeuralEstimators
using NeuralEstimatorsGNN
using GraphNeuralNetworks
using CSV
using DataFrames

include(joinpath(pwd(), "src/$model/Parameters.jl"))
include(joinpath(pwd(), "src/$model/Simulation.jl"))
include(joinpath(pwd(), "src/Architecture.jl"))

path = "intermediates/experiments/GNNarchitectures/$model"
if !isdir(path) mkpath(path) end


# --------------------
# ---- Experiment ----
# --------------------

# Factorial experiment to investigate key considerations different
# GNN architectures. In the propagation module, we consider the class of
# propagation layer; its complexity in terms of the number of channels in each
# layer; and the effect of local pooling between layers. We consider several
# readout modules, namely, mean, attention, and universal pooling.
# We assess the estimators in terms of training time and the estimated risk
# function.

# Main results:


# ---- Generate training and validation sets ----

# Size of the training, validation, and test sets
K_train = 10_000
K_val   = K_train ÷ 5
K_test  = K_val
if quick
	K_train = K_train ÷ 100
	K_val   = K_val ÷ 100
	K_test  = K_test ÷ 100
end

p = ξ.p
n = size(ξ.D, 1)

seed!(1)
θ_val   = Parameters(ξ, K_val,   J = 10)
θ_train = Parameters(ξ, K_train, J = 10)
Z_val   = [simulate(θ_val, mᵢ)   for mᵢ ∈ m]
Z_train = [simulate(θ_train, mᵢ) for mᵢ ∈ m]

# convert to graph
A = adjacencymatrix(ξ.D, ξ.r)
g = GNNGraph(A)
Z_train = reshapedataGNN.(Z_train, Ref(g))
Z_val   = reshapedataGNN.(Z_val, Ref(g))


# ---- set up ----

all_nh = quick ? [8, 16] : [8, 16, 32, 64]
all_globalpool = ["mean", "attention", "deepset"]
all_combinations = expandgrid(all_globalpool, all_nh)
estimator_names = ["$(c[1])pool_nh$(c[2])" for c ∈ eachrow(all_combinations)]

# ---- train the estimators  ----

times = []
estimators = map(eachindex(estimator_names)) do i
	c = all_combinations[i, :]
	θ̂ = gnnarchitecture(p, nh = c[2], globalpool = c[1])
	t = @elapsed train(θ̂, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_$(estimator_names[i])", epochs = quick ? 3 : 100)
	push!(times, t)
	Flux.loadparams!(θ̂, loadbestweights(path * "/runs_$(estimator_names[i])_m$M"))
	θ̂
end
CSV.write(path * "/traintime.csv", DataFrame(hcat(times, estimator_names), [:time, :estimator]))


#  ---- assess the estimators  ----

θ = Parameters(ξ, K_test)
Z = simulate(θ, M)
Z = reshapedataGNN(Z, g)
assessment = assess(estimators, θ, Z; estimator_names = estimator_names, parameter_names = ξ.parameter_names)
CSV.write(path * "/estimates.csv", assessment.df)
CSV.write(path * "/runtime.csv", assessment.runtime)
