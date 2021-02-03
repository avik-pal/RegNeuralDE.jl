struct TrackedNeuralDSDE{R,Z,M1,M2,P,RE1,RE2,T,A,K} <: DiffEqFlux.NeuralDELayer
    model1::M1
    model2::M2
    p::P
    re1::RE1
    re2::RE2
    tspan::T
    args::A
    kwargs::K
    len::Int
    nfes::Vector{Int}

    function TrackedNeuralDSDE(model1, model2, tspan, regularize, args...; kwargs...)
        return_multiple = get(kwargs, :save_everystep, false) || haskey(kwargs, :saveat)
        p1, re1 = Flux.destructure(model1)
        p2, re2 = Flux.destructure(model2)
        p = vcat(p1, p2)
        new{
            regularize,
            return_multiple,
            typeof(model1),
            typeof(model2),
            typeof(p),
            typeof(re1),
            typeof(re2),
            typeof(tspan),
            typeof(args),
            typeof(kwargs),
        }(
            model1,
            model2,
            p,
            re1,
            re2,
            tspan,
            args,
            kwargs,
            length(p1),
            [0, 0],
        )
    end
end

@fastmath function (n::TrackedNeuralDSDE{false,true})(x, p = n.p; func = (u, t, int) -> 0)
    function dudt_(u, p, t)
        n.nfes[1] += 1
        n.re1(p[1:n.len])(u)
    end
    function g(u, p, t)
        n.nfes[2] += 1
        n.re2(p[n.len+1:end])(u)
    end
    tspan = _convert_tspan(n.tspan, p)
    ff = SDEFunction{false}(dudt_, g, tgrad = DiffEqFlux.basic_tgrad)
    prob = SDEProblem{false}(ff, g, x, tspan, p)
    sol = solve(prob, n.args...; sensealg = SensitivityADPassThrough(), n.kwargs...)
    arr = diffeqsol_to_3dtrackedarray(sol)::TrackedArray{Float32,3,Array{Float32,3}}
    nfe1, nfe2 = n.nfes
    n.nfes .= 0

    return arr, nfe1, nfe2, nothing
end

@fastmath function (n::TrackedNeuralDSDE{false,false})(x, p = n.p; func = (u, t, int) -> 0)
    function dudt_(u, p, t)
        n.nfes[1] += 1
        n.re1(p[1:n.len])(u)
    end
    function g(u, p, t)
        n.nfes[2] += 1
        n.re2(p[n.len+1:end])(u)
    end
    tspan = _convert_tspan(n.tspan, p)
    ff = SDEFunction{false}(dudt_, g, tgrad = DiffEqFlux.basic_tgrad)
    prob = SDEProblem{false}(ff, g, x, tspan, p)
    sol = solve(prob, n.args...; sensealg = SensitivityADPassThrough(), n.kwargs...)
    arr = diffeqsol_to_trackedarray(sol)::TrackedArray{Float32,2,Array{Float32,2}}
    nfe1, nfe2 = n.nfes
    n.nfes .= 0

    return arr, nfe1, nfe2, nothing
end

@fastmath function (n::TrackedNeuralDSDE{true,true})(
    x,
    p = n.p;
    func = (u, t, integrator) -> integrator.EEst * integrator.dt,
)
    function dudt_(u, p, t)
        n.nfes[1] += 1
        n.re1(p[1:n.len])(u)
    end
    function g(u, p, t)
        n.nfes[2] += 1
        n.re2(p[n.len+1:end])(u)
    end
    tspan = _convert_tspan(n.tspan, p)
    sv = SavedValues(eltype(tspan), eltype(p))
    svcb = SavingCallback(func, sv)
    ff = SDEFunction{false}(dudt_, g, tgrad = DiffEqFlux.basic_tgrad)
    prob = SDEProblem{false}(ff, g, x, tspan, p)
    sol = solve(
        prob,
        n.args...;
        sensealg = SensitivityADPassThrough(),
        callback = svcb,
        n.kwargs...,
    )
    arr = diffeqsol_to_3dtrackedarray(sol)::TrackedArray{Float32,3,Array{Float32,3}}
    nfe1, nfe2 = n.nfes
    n.nfes .= 0

    return arr, nfe1, nfe2, sv
end

@fastmath function (n::TrackedNeuralDSDE{true,false})(
    x,
    p = n.p;
    func = (u, t, integrator) -> integrator.EEst * integrator.dt,
)
    function dudt_(u, p, t)
        n.nfes[1] += 1
        n.re1(p[1:n.len])(u)
    end
    function g(u, p, t)
        n.nfes[2] += 1
        n.re2(p[n.len+1:end])(u)
    end
    tspan = _convert_tspan(n.tspan, p)
    sv = SavedValues(eltype(tspan), eltype(p))
    svcb = SavingCallback(func, sv)
    ff = SDEFunction{false}(dudt_, g, tgrad = DiffEqFlux.basic_tgrad)
    prob = SDEProblem{false}(ff, g, x, tspan, p)
    sol = solve(
        prob,
        n.args...;
        sensealg = SensitivityADPassThrough(),
        callback = svcb,
        n.kwargs...,
    )
    arr = diffeqsol_to_trackedarray(sol)::TrackedArray{Float32,2,Array{Float32,2}}
    nfe1, nfe2 = n.nfes
    n.nfes .= 0

    return arr, nfe1, nfe2, sv
end
