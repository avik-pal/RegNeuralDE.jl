struct TrackedNeuralODE{R,Z,M,P,RE,T,A,K} <: DiffEqFlux.NeuralDELayer
    model::M
    p::P
    re::RE
    tspan::T
    args::A
    kwargs::K
    time_dep::Bool

    function TrackedNeuralODE(model, tspan, time_dep, regularize, args...; kwargs...)
        return_multiple = get(kwargs, :save_everystep, false) || haskey(kwargs, :saveat)
        p, re = Flux.destructure(model)
        kwargs = Dict(kwargs)
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

function update_saveat!(kwargs, saveat)
    if isnothing(saveat)
        return kwargs, () -> nothing
    end
    original_saveat = get(kwargs, :saveat, nothing)
    function restore!()
        kwargs[:saveat] = original_saveat
        return kwargs
    end
    kwargs[:saveat] = saveat
    return kwargs, restore!
end

@fastmath function (n::TrackedNeuralODE{false,false})(
    x,
    p = n.p;
    func = (u, t, int) -> 0,
    tspan = nothing,
    saveat = nothing,
)
    dudt_(u, p, t) = n.time_dep ? n.re(p)(u, t) : n.re(p)(u)

    tspan = _convert_tspan(isnothing(tspan) ? n.tspan : tspan, p)

    kwargs, restore_kwargs! = update_saveat!(n.kwargs, saveat)

    ff = ODEFunction{false}(dudt_, tgrad = DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, tspan, p)

    sol = solve(
        prob,
        n.args...;
        sensealg = SensitivityADPassThrough(),
        callback = nothing,
        kwargs...,
    )
    res = diffeqsol_to_trackedarray(sol)::typeof(x)
    nfe = sol.destats.nf::Int

    restore_kwargs!()

    return res, nfe, nothing
end

@fastmath function (n::TrackedNeuralODE{false,true})(
    x,
    p = n.p;
    func = (u, t, int) -> 0,
    tspan = nothing,
    saveat = nothing,
)
    dudt_(u, p, t) = n.time_dep ? n.re(p)(u, t) : n.re(p)(u)

    tspan = _convert_tspan(isnothing(tspan) ? n.tspan : tspan, p)

    kwargs, restore_kwargs! = update_saveat!(n.kwargs, saveat)

    ff = ODEFunction{false}(dudt_, tgrad = DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, tspan, p)

    sol = solve(
        prob,
        n.args...;
        sensealg = SensitivityADPassThrough(),
        callback = nothing,
        kwargs...,
    )
    res = diffeqsol_to_3dtrackedarray(sol)::TrackedArray{Float32,3,CuArray{Float32,3}}
    nfe = sol.destats.nf::Int

    restore_kwargs!()

    return res, nfe, nothing
end

@fastmath function (n::TrackedNeuralODE{true,false})(
    x,
    p = n.p;
    # Default is to regularize using Error Estimates. Alternative tested
    # strategy is using Stiffness Estimate:
    # (u, t, integrator) -> integrator.eigen_est * integrator.dt
    func = (u, t, integrator) -> integrator.EEst * integrator.dt,
    tspan = nothing,
    saveat = nothing,
)
    dudt_(u, p, t) = n.time_dep ? n.re(p)(u, t) : n.re(p)(u)

    tspan = _convert_tspan(isnothing(tspan) ? n.tspan : tspan, p)

    kwargs, restore_kwargs! = update_saveat!(n.kwargs, saveat)

    sv = SavedValues(eltype(tspan), eltype(p))
    svcb = SavingCallback(func, sv)
    ff = ODEFunction{false}(dudt_, tgrad = DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, tspan, p)

    sol = solve(
        prob,
        n.args...;
        sensealg = SensitivityADPassThrough(),
        callback = svcb,
        kwargs...,
    )
    res = diffeqsol_to_trackedarray(sol)::typeof(x)

    restore_kwargs!()

    nfe = sol.destats.nf::Int
    return res, nfe, sv
end

@fastmath function (n::TrackedNeuralODE{true,true})(
    x,
    p = n.p;
    # Default is to regularize using Error Estimates. Alternative tested
    # strategy is using Stiffness Estimate:
    # (u, t, integrator) -> integrator.eigen_est * integrator.dt
    func = (u, t, integrator) -> integrator.EEst * integrator.dt,
    tspan = nothing,
    saveat = nothing,
)
    dudt_(u, p, t) = n.time_dep ? n.re(p)(u, t) : n.re(p)(u)

    tspan = _convert_tspan(isnothing(tspan) ? n.tspan : tspan, p)

    kwargs, restore_kwargs! = update_saveat!(n.kwargs, saveat)

    sv = SavedValues(eltype(tspan), eltype(p))
    svcb = SavingCallback(func, sv)
    ff = ODEFunction{false}(dudt_, tgrad = n.time_dep ? nothing : DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, tspan, p)

    sol = solve(
        prob,
        n.args...;
        sensealg = SensitivityADPassThrough(),
        callback = svcb,
        kwargs...,
    )
    res = diffeqsol_to_3dtrackedarray(sol)::TrackedArray{Float32,3,CuArray{Float32,3}}

    restore_kwargs!()

    nfe = sol.destats.nf::Int
    return res, nfe, sv
end

function solution(
    n::TrackedNeuralODE,
    x,
    p = n.p;
    solver = nothing,
    tspan = nothing,
    saveat = nothing,
)
    solver = isnothing(solver_override) ? n.args[1] : solver
    dudt_(u, p, t) = n.time_dep ? n.re(p)(u, t) : n.re(p)(u)

    tspan = _convert_tspan(isnothing(tspan) ? n.tspan : tspan, p)

    kwargs, restore_kwargs! = update_saveat!(n.kwargs, saveat)

    ff = ODEFunction{false}(dudt_, tgrad = n.time_dep ? nothing : DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, tspan, p)

    sol = solve(
        prob,
        solver;
        sensealg = SensitivityADPassThrough(),
        callback = nothing,
        kwargs...,
    )

    restore_kwargs!()

    return sol
end
