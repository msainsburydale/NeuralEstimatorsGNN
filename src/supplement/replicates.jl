# -----------------------------------------
# ---- Handling independent replicates ----
# -----------------------------------------

model="GP/nuFixed"
K = 100
m=[1, 30, 90]

using NeuralEstimators
using NeuralEstimatorsGNN
using DataFrames
using GraphNeuralNetworks
using CSV
using BenchmarkTools

# Prevent scalar indexing
using CUDA
CUDA.allowscalar(false)

include(joinpath(pwd(), "src/$model/model.jl"))
include(joinpath(pwd(), "src/$model/ML.jl"))
include(joinpath(pwd(), "src/architecture.jl"))

p = ξ.p
n = ξ.n

# ---- Setup ----

# Test on a grid so that we can compare to a CNN
pts = range(0, 1, length = isqrt(n))
S = expandgrid(pts, pts)
D = pairwise(Euclidean(), S, S, dims = 1)
ξ = (ξ..., D = D)
A = adjacencymatrix(ξ.D, ξ.r)
g = GNNGraph(A)

seed!(1)
cnn = DeepSet(cnnarchitecture(p)...)
dnn = DeepSet(dnnarchitecture(n, p)...)
gnn = gnnarchitecture(p; propagation = "GraphConv")

# Compare the number of trainable parameters
nparams(cnn)  # 636062
nparams(dnn)  # 238658
nparams(gnn)  # 181890

# Sample parameters and simulate training/validation data
seed!(1)
θ = Parameters(K, ξ)
Z = [simulate(θ, mᵢ) for mᵢ ∈ m];


# ---- Speed tests ----

# CNN (gold standard)
Z1 = Z[1]; # m=1 replicates per parameter configuration
Z2 = Z[2]; # m=30 replicates per parameter configuration
Z3 = Z[3]; # m=90 replicates per parameter configuration
Z1 = reshapedataCNN(Z1);
Z2 = reshapedataCNN(Z2);
Z3 = reshapedataCNN(Z3);
@btime cnn(Z1); # 17.1  ms (1761 allocations: 5.17 MiB)
@btime cnn(Z2); # 596   ms (1863 allocations: 114.20 MiB)
@btime cnn(Z3); # 1776  ms (1863 allocations: 339.78 MiB)
Z1  = Z1  |> gpu;
Z2  = Z2  |> gpu;
Z3  = Z3  |> gpu;
cnn = cnn |> gpu;
@btime cnn(Z1); # 3.327 ms (13758 allocations: 811.41 KiB)
@btime cnn(Z2); # 3.849 ms (13774 allocations: 3.62 MiB)
@btime cnn(Z3); # 4.842 ms (13766 allocations: 9.48 MiB)

# GNN: batching approach that does not exploit constant graph structure
Z1 = Z[1];
Z2 = Z[2];
Z3 = Z[3];
Z1 = reshapedataGNN(Z1, g);
Z2 = reshapedataGNN(Z2, g);
Z3 = reshapedataGNN(Z3, g);
@btime gnn(Z1);  # 306.044 ms (13011 allocations: 831.30 MiB)
@btime gnn(Z2);  # 11641 ms (13023 allocations: 24.31 GiB)
Z1  = Z1  |> gpu;
Z2  = Z2  |> gpu;
Z3  = Z3  |> gpu;
gnn = gnn |> gpu;
@btime gnn(Z1);  # 18.804 ms (37349 allocations: 2.28 MiB)
@btime gnn(Z2);  # 151.327 ms (37525 allocations: 3.70 MiB)
@btime gnn(Z3);  # 505.318 ms (37534 allocations: 6.63 MiB)

# GNN: approach that exploits constant graph structure with parameter configurations
Z1 = Z[1];
Z2 = Z[2];
Z3 = Z[3];
using NeuralEstimatorsGNN: reshapedataGNN2
Z1 = reshapedataGNN2(Z1, g);
Z2 = reshapedataGNN2(Z2, g);
Z3 = reshapedataGNN2(Z3, g);
Z1  = Z1  |> gpu;
Z2  = Z2  |> gpu;
Z3  = Z3  |> gpu;
gnn = gnn |> gpu;
@btime gnn(Z1); # 23.429 ms (38700 allocations: 2.34 MiB)
@btime gnn(Z2); # 218.167 ms (38781 allocations: 3.76 MiB)
@btime gnn(Z3); # 648.431 ms (39873 allocations: 6.73 MiB)

# GNN: approach that exploits constant graph structure for ALL replicates
Z1 = Z[1];
Z2 = Z[2];
Z3 = Z[3];
reshapedataGNN3(Z, g) = GNNGraphFixedStructure(g, addsingleton.(permutedims.(Z), dim = 1))
Z1 = reshapedataGNN3(Z1, g);
Z2 = reshapedataGNN3(Z2, g);
Z3 = reshapedataGNN3(Z3, g);

Z1  = Z1  |> gpu;
Z2  = Z2  |> gpu;
Z3  = Z3  |> gpu;
gnn = gnn |> gpu;

@btime gnn(Z1); # 11.515 ms (17001 allocations: 880.22 KiB)
@btime gnn(Z2); # 182.614 ms (17082 allocations: 2.28 MiB)
@btime gnn(Z3); # 624.765 ms (17375 allocations: 5.22 MiB)


# Summary:
# CNN scales very, very well on the GPU.
# The GNN does not scale so well.
