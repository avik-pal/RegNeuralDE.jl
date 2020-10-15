using RegNeuralODE, OrdinaryDiffEq, Flux, DiffEqFlux, Tracker, Random, Statistics
using ProgressLogging, YAML, Dates, BSON
using CUDA
using RegNeuralODE: accuracy
using Flux.Optimise: update!
using Flux: @functor, glorot_uniform, logitcrossentropy
using Tracker: TrackedReal, data
import Base.show


struct ConcatSquashLayer{L, B, G}
    linear::L
    hyper_bias::B
    hyper_gate::G
end

function ConcatSquashLayer(in_dim::Int, out_dim::Int)
    l = Dense(in_dim, out_dim)
    b = Linear(1, out_dim)
    g = Dense(1, out_dim)
    return ConcatSquashLayer(l, b, g)
end

@functor ConcatSquashLayer

function (csl::ConcatSquashLayer)(x, t)
    _t = reshape(Tracker.collect([t]), 1, 1)
    return csl.linear(x) .* σ.(csl.hyper_gate(_t)) .+ csl.hyper_bias(_t)
end


struct MLPDynamics{L1, L2, L3}
    l1::L1
    l2::L2
    l3::L3
end

MLPDynamics(in_dims::Int, hdim1::Int, hdim2::Int) =
    MLPDynamics(ConcatSquashLayer(in_dims, hdim1),
                ConcatSquashLayer(hdim1, hdim2),
                ConcatSquashLayer(hdim2, in_dims))

@functor MLPDynamics

function (mlp::MLPDynamics)(x, t)
    x = softplus.(mlp.l1(x, t))
    x = softplus.(mlp.l2(x, t))
    return mlp.l3(x, t)
end