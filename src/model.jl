# MLP Dynamics conditioned on the Time Step
struct TimeDependentMLPDynamics{M}
    model::M
    
    function TimeDependentMLPDynamics(dims::Vector{Int}, act = tanh)
        layers = []
        for i in 1:length(dims) - 2
            push!(layers, Dense(dims[i] + 1, dims[i + 1], act))
        end
        push!(layers, Dense(dims[end - 1] + 1, dims[end]))
        model = Chain(layers...)
        return new{typeof(model)}(model)
    end
end

Flux.@functor TimeDependentMLPDynamics

function (m::TimeDependentMLPDynamics)(x, t::Number)
    _t = similar(x, 1, size(x, 2))
    fill!(_t, t)
    for layer in m.model
        x = layer(cat(x, _t, dims = 1))
    end
    return x
end

# Neural ODE Variants
struct NFECounterNeuralODE{M, P, RE, T, A, K} <: DiffEqFlux.NeuralDELayer
    model::M
    p::P
    re::RE
    tspan::T
    args::A
    kwargs::K
    nfe::Vector{Int}

    function NFECounterNeuralODE(model, tspan, args...; p = nothing,
                                 kwargs...)
        _p, re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        new{typeof(model), typeof(p), typeof(re),
            typeof(tspan), typeof(args), typeof(kwargs)}(
            model, p, re, tspan, args, kwargs, [0])
    end
end

function (n::NFECounterNeuralODE)(x, p = n.p)
    function dudt_(u, p, t)
        n.nfe[] += 1
        n.re(p)(u)
    end
    ff = ODEFunction{false}(dudt_, tgrad = DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, n.tspan, p)
    solve(prob, n.args...; sensealg = SensitivityADPassThrough(), n.kwargs...)
end

function (n::NFECounterNeuralODE{M})(x, p = n.p) where M <: TimeDependentMLPDynamics
    function dudt_(u, p, t)
        n.nfe[] += 1
        n.re(p)(u, t)
    end
    ff = ODEFunction{false}(dudt_, tgrad = DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, n.tspan, p)
    solve(prob, n.args...; sensealg = SensitivityADPassThrough(), n.kwargs...)
end


struct NFECounterCallbackNeuralODE{M, P, RE, T, A, K} <: DiffEqFlux.NeuralDELayer
    model::M
    p::P
    re::RE
    tspan::T
    args::A
    kwargs::K
    nfe::Vector{Int}

    function NFECounterCallbackNeuralODE(model, tspan, args...; p = nothing,
                                         kwargs...)
        _p, re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        new{typeof(model), typeof(p), typeof(re),
            typeof(tspan), typeof(args), typeof(kwargs)}(
            model, p, re, tspan, args, kwargs, [0])
    end
end

function (n::NFECounterCallbackNeuralODE)(x, p::AbstractArray{T} = n.p) where T
    function dudt_(u, p, t)
        n.nfe[] += 1
        n.re(p)(u)
    end

    tspan = T.(n.tspan)

    sv = SavedValues(eltype(tspan), T)
    svcb = SavingCallback(
        (u, t, integrator) -> integrator.EEst * integrator.dt, sv
    )

    ff = ODEFunction{false}(dudt_, tgrad=DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, tspan, p)

    solve(prob, n.args...; sensealg = SensitivityADPassThrough(),
          callback = svcb, n.kwargs...), sv
end

function (n::NFECounterCallbackNeuralODE{M})(x, p::AbstractArray{T} = n.p) where {T, M <: TimeDependentMLPDynamics}
    function dudt_(u, p, t)
        n.nfe[] += 1
        n.re(p)(u, t)
    end

    tspan = T.(n.tspan)

    sv = SavedValues(eltype(tspan), T)
    svcb = SavingCallback(
        (u, t, integrator) -> integrator.EEst * integrator.dt, sv
    )

    ff = ODEFunction{false}(dudt_, tgrad=DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, tspan, p)

    solve(prob, n.args...; sensealg = SensitivityADPassThrough(),
          callback = svcb, n.kwargs...), sv
end


# Classification Network mostly useful for ReverseDiff
struct ClassifierNODE{N, P1, P2, T}
    preode::P1
    node::N
    postode::P2
    p1::T
    p2::T
    p3::T
    
    function ClassifierNODE(preode, node, postode)
        p1, re1 = Flux.destructure(preode)
        p2 = node.p
        p3, re3 = Flux.destructure(postode)
        return new{typeof(node), typeof(re1), typeof(re3),
                   typeof(p2)}(re1, node, re3, p1, p2, p3)
    end
end

Flux.trainable(m::ClassifierNODE) = (m.p1, m.p2, m.p3)

function (m::ClassifierNODE)(x, p1 = m.p1, p2 = m.p2, p3 = m.p3)
    x = m.preode(p1)(x)
    x = m.node(x, p2)
    return m.postode(p3)(x)
end

function (m::ClassifierNODE{T})(x, p1 = m.p1, p2 = m.p2,
                                p3 = m.p3) where T<:NFECounterCallbackNeuralODE
    x = m.preode(p1)(x)
    x, sv = m.node(x, p2)
    return m.postode(p3)(x), sv
end
