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
using ReverseDiff, ForwardDiff, Tracker, Zygote
using Tracker: data, TrackedReal
# Plotting
using Plots
# Data Processing
using MLDatasets, MLDataUtils, BSON
using Flux.Data: DataLoader
# Helper Modules
using MacroTools: @forward

# Hacks to make things work
function (ReverseDiff.TrackedReal{V, D, O})(val::ForwardDiff.Dual) where {V, D, O}
    ReverseDiff.TrackedReal(val.value, val.partials.values[1])
end

Base.nextfloat(x::ReverseDiff.TrackedReal) = x


# These should be in Tracker.jl
## Track the parameters
track(m) = fmap(x -> x isa AbstractArray ? Tracker.param(x) : x, m)
untrack(m) = fmap(Tracker.data, m)


# Include code
include("dataset.jl")
include("model.jl")
include("utils.jl")
include("train.jl")


# Export functions
export load_mnist, load_physionet, load_spiral2d
export NFECounterCallbackNeuralODE, NFECounterNeuralODE, ClassifierNODE,
       TDChain, RecognitionRNN, ExtrapolationLatentODE, Linear, NFECounterFFJORD,
       NFECounterCallbackFFJORD
export track, untrack

end
