module RegNeuralODE

# StdLib
using Statistics, LinearAlgebra, Printf
# DiffEq Packages
using DiffEqFlux, OrdinaryDiffEq, DiffEqCallbacks, DiffEqSensitivity
# Neural Networks
using Flux
# AD Packages
using ReverseDiff, ForwardDiff
# Plotting
using Plots
# Data Processing
using MLDatasets, MLDataUtils, BSON
using Flux.Data: DataLoader


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
export load_mnist, load_physionet
export NFECounterCallbackNeuralODE, NFECounterNeuralODE, ClassifierNODE, TimeDependentMLPDynamics

end
