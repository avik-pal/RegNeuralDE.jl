module RegNeuralODE

# StdLib
using Statistics, LinearAlgebra, Printf
# DiffEq Packages
using DiffEqFlux, OrdinaryDiffEq, DiffEqCallbacks, DiffEqSensitivity
# Neural Networks
using Flux, CUDA
# AD Packages
using ReverseDiff, ForwardDiff, Tracker
# Plotting
using Plots
# Data Processing
using MLDatasets, MLDataUtils, BSON
using Flux.Data: DataLoader

# The latest version of Distributions AD has these fixes. For now keeping
# them here
Base.prevfloat(r::Tracker.TrackedReal) = Tracker.track(prevfloat, r)
Tracker.@grad function prevfloat(r::Real)
    prevfloat(Tracker.data(r)), Δ -> (Δ,)
end
Base.nextfloat(r::Tracker.TrackedReal) = Tracker.track(nextfloat, r)
Tracker.@grad function nextfloat(r::Real)
    nextfloat(Tracker.data(r)), Δ -> (Δ,)
end

# Hacks to make things work
function (ReverseDiff.TrackedReal{V, D, O})(val::ForwardDiff.Dual) where {V, D, O}
    ReverseDiff.TrackedReal(val.value, val.partials.values[1])
end

Base.nextfloat(x::ReverseDiff.TrackedReal) = x


# Include code
include("dataset.jl")
include("model.jl")
include("utils.jl")
include("train.jl")


# Export functions
export load_mnist, load_physionet, load_spiral2d
export NFECounterCallbackNeuralODE, NFECounterNeuralODE, ClassifierNODE,
       TimeDependentMLPDynamics, RecognitionRNN, ExtrapolationLatentODE

end
