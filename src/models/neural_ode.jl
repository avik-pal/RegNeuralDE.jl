struct TrackedNeuralODE{R,Z,M,P,RE,T,A,K} <: DiffEqFlux.NeuralDELayer
    model::M
    p::P
    re::RE
    tspan::T
    args::A
    kwargs::K
    time_dep::Bool

    function TrackedNeuralODE(model, tspan, time_dep, regularize, args...; kwargs...)
        return_multiple =
            hasproperty(kwargs, :save_everystep) ? kwargs.save_everystep :
            hasproperty(kwargs, :saveat)
        p, re = Flux.destructure(model)
        new{
            regularize,
            return_multiple,
            typeof(model),
            typeof(p),
            typeof(re),
            typeof(tspan),
            typeof(args),
            typeof(kwargs),
        }(
            model,
            p,
            re,
            tspan,
            args,
            kwargs,
            time_dep,
        )
    end
end

function (n::TrackedNeuralODE{false,false})(x, p = n.p; func = (u, t, int) -> 0)
    dudt_(u, p, t) = n.time_dep ? n.re(p)(u, t) : n.re(p)(u)

    tspan = _convert_tspan(n.tspan, p)

    ff = ODEFunction{false}(dudt_, tgrad = DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, tspan, p)

    sol = solve(
        prob,
        n.args...;
        sensealg = SensitivityADPassThrough(),
        callback = nothing,
        n.kwargs...,
    )
    res = diffeqsol_to_trackedarray(sol)::typeof(x)
    nfe = sol.destats.nf::Int

    return res, nfe, nothing
end

function (n::TrackedNeuralODE{false,true})(x, p = n.p; func = (u, t, int) -> 0)
    dudt_(u, p, t) = n.time_dep ? n.re(p)(u, t) : n.re(p)(u)

    tspan = _convert_tspan(n.tspan, p)

    ff = ODEFunction{false}(dudt_, tgrad = DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, tspan, p)

    sol = solve(
        prob,
        n.args...;
        sensealg = SensitivityADPassThrough(),
        callback = nothing,
        n.kwargs...,
    )
    res = diffeqsol_to_3dtrackedarray(sol)::TrackedArray{Float32,3,CuArray{Float32,3}}
    nfe = sol.destats.nf::Int

    return res, nfe, nothing
end

function (n::TrackedNeuralODE{true,false})(
    x,
    p = n.p;
    # Default is to regularize using Error Estimates. Alternative tested
    # strategy is using Stiffness Estimate:
    # (u, t, integrator) -> integrator.eigen_est * integrator.dt
    func = (u, t, integrator) -> integrator.EEst * integrator.dt,
)
    dudt_(u, p, t) = n.time_dep ? n.re(p)(u, t) : n.re(p)(u)

    tspan = _convert_tspan(n.tspan, p)

    sv = SavedValues(eltype(tspan), eltype(p))
    svcb = SavingCallback(func, sv)
    ff = ODEFunction{false}(dudt_, tgrad = DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, tspan, p)

    sol = solve(
        prob,
        n.args...;
        sensealg = SensitivityADPassThrough(),
        callback = svcb,
        n.kwargs...,
    )
    res = diffeqsol_to_trackedarray(sol)::typeof(x)

    nfe = sol.destats.nf::Int
    return res, nfe, sv
end

function (n::TrackedNeuralODE{true,true})(
    x,
    p = n.p;
    # Default is to regularize using Error Estimates. Alternative tested
    # strategy is using Stiffness Estimate:
    # (u, t, integrator) -> integrator.eigen_est * integrator.dt
    func = (u, t, integrator) -> integrator.EEst * integrator.dt,
)
    dudt_(u, p, t) = n.time_dep ? n.re(p)(u, t) : n.re(p)(u)

    tspan = _convert_tspan(n.tspan, p)

    sv = SavedValues(eltype(tspan), eltype(p))
    svcb = SavingCallback(func, sv)
    ff = ODEFunction{false}(dudt_, tgrad = DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, tspan, p)

    sol = solve(
        prob,
        n.args...;
        sensealg = SensitivityADPassThrough(),
        callback = svcb,
        n.kwargs...,
    )
    res = diffeqsol_to_3dtrackedarray(sol)::TrackedArray{Float32,3,CuArray{Float32,3}}

    nfe = sol.destats.nf::Int
    return res, nfe, sv
end

function solution(n::TrackedNeuralODE, x, p = n.p)
    dudt_(u, p, t) = n.time_dep ? n.re(p)(u, t) : n.re(p)(u)

    tspan = _convert_tspan(n.tspan, p)

    ff = ODEFunction{false}(dudt_, tgrad = DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, tspan, p)

    sol = solve(
        prob,
        n.args...;
        sensealg = SensitivityADPassThrough(),
        callback = nothing,
        n.kwargs...,
    )

    return sol
end
