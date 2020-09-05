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

function _get_dudt(n::NFECounterNeuralODE)
    function dudt_(u, p, t)
        n.nfe[] += 1
        n.re(p)(u)
    end
end

function _get_dudt(n::NFECounterNeuralODE{M}) where M <: TDChain
    function dudt_(u, p, t)
        n.nfe[] += 1
        n.re(p)(u, t)
    end
end

function (n::NFECounterNeuralODE)(x, p = n.p)
    dudt_ = _get_dudt(n)
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

_convert_tspan(tspan, p) = eltype(p).(tspan)

_convert_tspan(tspan, p::TrackedArray) = Tracker.collect(eltype(p).(tspan))

function _get_dudt(n::NFECounterCallbackNeuralODE)
    function dudt_(u, p, t)
        n.nfe[] += 1
        n.re(p)(u)
    end
end

function _get_dudt(n::NFECounterCallbackNeuralODE{M}) where M <: TDChain
    function dudt_(u, p, t)
        n.nfe[] += 1
        n.re(p)(u, t)
    end
end

function (n::NFECounterCallbackNeuralODE)(x, p = n.p)
    dudt_ = _get_dudt(n)
    tspan = _convert_tspan(n.tspan, p)

    sv = SavedValues(eltype(tspan), eltype(p))
    svcb = SavingCallback(
        (u, t, integrator) -> integrator.EEst * integrator.dt, sv
    )

    ff = ODEFunction{false}(dudt_, tgrad = DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, tspan, p)

    solve(prob, n.args...; sensealg = SensitivityADPassThrough(),
          callback = svcb, n.kwargs...), sv
end