struct TrackedNeuralODE{R, M, P, RE, T, A, K} <: DiffEqFlux.NeuralDELayer
    model::M
    p::P
    re::RE
    tspan::T
    args::A
    kwargs::K
    time_dep::Bool

    function TrackedNeuralODE(model, tspan, time_dep, regularize,
                              args...; kwargs...)
        p, re = Flux.destructure(model)
        new{regularize, typeof(model), typeof(p), typeof(re),
            typeof(tspan), typeof(args), typeof(kwargs)}(
            model, p, re, tspan, args, kwargs, time_dep)
    end
end

function (n::TrackedNeuralODE{false})(x, p = n.p)
    dudt_(u, p, t) = n.time_dep ? n.re(p)(u, t) : n.re(p)(u)

    tspan = _convert_tspan(n.tspan, p)

    ff = ODEFunction{false}(dudt_, tgrad = DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, tspan, p)

    sol = solve(prob, n.args...; sensealg = SensitivityADPassThrough(),
                callback = nothing, n.kwargs...)
    
    # cat doesn't preserve types
    # res = diffeqsol_to_trackedarray(sol) :: TrackedArray{Float32, 3, CuArray{Float32, 3}}
    res = diffeqsol_to_trackedarray(sol) :: typeof(x)
    nfe = sol.destats.nf :: Int

    return res, nfe, nothing
end


function (n::TrackedNeuralODE{true})(x, p = n.p)
    dudt_(u, p, t) = n.time_dep ? n.re(p)(u, t) : n.re(p)(u)

    tspan = _convert_tspan(n.tspan, p)

    sv = SavedValues(eltype(tspan), eltype(p))
    svcb = SavingCallback(
        (u, t, integrator) -> integrator.EEst * integrator.dt, sv
    )
    ff = ODEFunction{false}(dudt_, tgrad = DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, tspan, p)

    sol = solve(prob, n.args...; sensealg = SensitivityADPassThrough(),
                callback = svcb, n.kwargs...)

    # cat doesn't preserve types
    # res = diffeqsol_to_trackedarray(sol) :: TrackedArray{Float32, 3, CuArray{Float32, 3}}
    res = diffeqsol_to_trackedarray(sol) :: typeof(x)
    nfe = sol.destats.nf :: Int

    return res, nfe, sv
end