struct TrackedFFJORD{R,M,P,RE,D,T,A,K} <: DiffEqFlux.CNFLayer
    model::M
    p::P
    re::RE
    basedist::D
    tspan::T
    args::A
    kwargs::K

    function TrackedFFJORD(
        model,
        tspan,
        regularize,
        in_dims,
        args...;
        basedist = nothing,
        kwargs...,
    )
        p, re = Flux.destructure(model)
        if basedist === nothing
            size_input = in_dims
            T = Float32
            basedist = MvNormal(
                zeros(Float32, size_input),
                I + zeros(Float32, size_input, size_input),
            )
        end
        new{
            regularize,
            typeof(model),
            typeof(p),
            typeof(re),
            typeof(basedist),
            typeof(tspan),
            typeof(args),
            typeof(kwargs),
        }(
            model,
            p,
            re,
            basedist,
            tspan,
            args,
            kwargs,
        )
    end
end


# This regularize corresponds to the regularization proposed in the original
# paper
function _ffjord(u, p, t, re, e, regularize, M)
    m = re(p)::M
    if regularize
        z = u[1:end-3, :]
        mz, back = Tracker.forward(m, z, t)
        eJ = back(e)[1]
        trace_jac = sum(eJ .* e, dims = 1)
        return vcat(mz, -trace_jac, sum(abs2.(mz), dims = 1), norm_batched(eJ) .^ 2)
    else
        z = u[1:end-1, :]
        mz, back = Tracker.forward(m, z, t)
        eJ = back(e)[1]
        trace_jac = sum(eJ .* e, dims = 1)
        return vcat(mz, -trace_jac)
    end
end

function (n::TrackedFFJORD{false,M})(
    x,
    p = n.p,
    e = TrackedArray(CUDA.randn(Float32, size(x)));
    regularize = false,
) where {M}
    pz = n.basedist
    sense = SensitivityADPassThrough()
    tspan = _convert_tspan(n.tspan, p)
    ffjord_ = (u, p, t) -> _ffjord(u, p, t, n.re, e, regularize, M)
    if regularize
        _z = TrackedArray(CUDA.zeros(Float32, 3, size(x, 2)))

        prob = ODEProblem{false}(ffjord_, vcat(x, _z), tspan, p)
        sol = solve(prob, n.args...; sensealg = sense, n.kwargs...)

        pred = sol.u[1]::TrackedArray{Float32,2,CuArray{Float32,2}}
        z = pred[1:end-3, :]
        delta_logp = pred[end-2:end-2, :]
        λ₁ = pred[end-1, :]
        λ₂ = pred[end, :]
    else
        _z = TrackedArray(CUDA.zeros(Float32, 1, size(x, 2)))

        prob = ODEProblem{false}(ffjord_, vcat(x, _z), tspan, p)
        sol = solve(prob, n.args...; sensealg = sense, n.kwargs...)

        pred = sol.u[1]::TrackedArray{Float32,2,CuArray{Float32,2}}
        z = pred[1:end-1, :]
        delta_logp = pred[end:end, :]
        λ₁ = λ₂ = _z[1, :]
    end

    # return z, nothing

    # logpdf promotes the type to Float64 by default
    # This function is type unstable when used with Tracker
    logpz =
        (
            reshape(logpdf(pz, z |> cpu), 1, :) |> gpu
        )::TrackedArray{Float32,2,CuArray{Float32,2}}
    logpx = logpz .- delta_logp

    return logpx, λ₁, λ₂, sol.destats.nf
end

function (n::TrackedFFJORD{true})(
    x,
    p = n.p,
    e = Tracker.collect(randn(eltype(x), size(x)));
    regularize = false,
)
    pz = n.basedist
    tspan = _convert_tspan(n.tspan, p)
    sense = SensitivityADPassThrough()
    sv = SavedValues(eltype(tspan), eltype(p))
    svcb = SavingCallback((u, t, integrator) -> integrator.EEst * integrator.dt, sv)
    ffjord_ = (u, p, t) -> _ffjord(u, p, t, n.re, e, false)
    _z = CUDA.zeros(Float32, 1, size(x, 2)) |> track

    prob = ODEProblem{false}(ffjord_, vcat(x, _z), tspan, p)
    sol = solve(prob, n.args...; sensealg = sense, callback = svcb, n.kwargs...)
    pred = sol.u[1]
    z = pred[1:end-1, :]
    delta_logp = pred[end:end, :]

    # logpdf promotes the type to Float64 by default
    # This function is type unstable when used with Tracker
    logpz = reshape(logpdf(pz, z |> cpu), 1, :) |> gpu
    logpx = logpz .- delta_logp

    return logpx, sv, sol.destats.nf
end
