module RegNeuralODE

# StdLib
using Statistics, LinearAlgebra, Printf
# DiffEq Packages
using DiffEqFlux, OrdinaryDiffEq, DiffEqCallbacks, DiffEqSensitivity
# Neural Networks
using Flux, CUDA
# Probabilistic Stuff
using Distributions, DistributionsAD
# AD Packages
using Tracker
using Tracker: data, TrackedReal
# Data Processing
using MLDatasets, MLDataUtils, BSON
using Flux.Data: DataLoader
# Helper Modules
using MacroTools: @forward


# These should be in Tracker.jl
## Track the parameters
track(m) = fmap(x -> x isa AbstractArray ? Tracker.param(x) : x, m)
untrack(m) = fmap(Tracker.data, m)


# Include code
include("dataset.jl")
include("utils.jl")
include("model.jl")
include("metrics.jl")
include("losses.jl")


# Export functions
export load_mnist, load_physionet, load_spiral2d
export TrackedNeuralODE, ClassifierNODE, TDChain, RecognitionRNN, Linear,
       TrackedFFJORD, PhysionetLatentGRU
export track, untrack

end
