module RegNeuralODE

# StdLib
using Statistics, LinearAlgebra
# DiffEq Packages
using DiffEqFlux, OrdinaryDiffEq, DiffEqCallbacks, DiffEqSensitivity
# Neural Networks
using Flux, CUDA
import Flux.Optimise.update!
# Probabilistic Stuff
using Distributions, DistributionsAD
# AD Packages
using Tracker
using Tracker: data, TrackedReal
# Data Processing
using MLDatasets, MLDataUtils, BSON, NPZ
using Flux.Data: DataLoader
# Helper Modules
using MacroTools: @forward
using Format


# These should be in Tracker.jl
## Track the parameters
track(m) = fmap(x -> x isa AbstractArray ? Tracker.param(x) : x, m)
untrack(m) = fmap(Tracker.data, m)

# Hack to get around scalar indexing issue in gradients for FFJORD
# See https://github.com/JuliaGPU/Adapt.jl/issues/21
Base.convert(
    ::Type{CuArray{Float32,2}},
    x::Base.ReshapedArray{
        Float32,
        2,
        Transpose{Float32,CuArray{Float32,2}},
        Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}},
    },
) = CuArray(x)

# Include code
include("dataset.jl")
include("utils.jl")
include("models/basic.jl")
include("models/neural_ode.jl")
include("models/supervised_classification.jl")
include("models/time_series.jl")
include("models/ffjord.jl")
include("metrics.jl")


# Export functions
export load_mnist, load_physionet, load_spiral2d, load_miniboone, load_multimodel_gaussian
export TrackedNeuralODE,
    ClassifierNODE, TDChain, RecognitionRNN, TrackedFFJORD, LatentTimeSeriesModel
export track, untrack, table_logger, solution, sample, update_parameters!

end
