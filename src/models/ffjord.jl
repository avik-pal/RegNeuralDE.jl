struct TrackedFFJORD{R,M,P,RE,T,A,K} <: DiffEqFlux.CNFLayer
    model::M
    p::P
    re::RE
    tspan::T
    args::A
    kwargs::K

    function TrackedFFJORD(model, tspan, regularize, args...; kwargs...)
        p, re = Flux.destructure(model)
        new{
            regularize,
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
        )
    end
end

@fastmath function _ffjord(u, p, t, re, e, regularize, M)
    m = re(p)::M
    if regularize
        z = u[1:end-3, :] |> untrack
        mz, back = Tracker.forward(z -> m(z, t), z)
        eJ = back(e)[1]
        trace_jac = sum(eJ .* e, dims = 1)
        return vcat(mz, -trace_jac, sum(abs2.(mz), dims = 1), norm_batched(eJ) .^ 2)
    else
        z = u[1:end-1, :] |> untrack
        mz, back = Tracker.forward(z -> m(z, t), z)
        eJ = back(e)[1]
        trace_jac = sum(eJ .* e, dims = 1)
        return vcat(mz, -trace_jac)
    end
end

@fastmath function (n::TrackedFFJORD{false,M})(
    x,
    p = n.p,
    e = CUDA.randn(Float32, size(x)...);
    regularize = false,
) where {M}
    sense = SensitivityADPassThrough()
    tspan = _convert_tspan(n.tspan, p)
    ffjord_ = (u, p, t) -> _ffjord(u, p, t, n.re, e, regularize, M)
    if regularize
        _z = TrackedArray(CUDA.zeros(Float32, 3, size(x, 2)))

        prob = ODEProblem{false}(ffjord_, vcat(x, _z), tspan, p)
        sol = solve(prob, n.args...; sensealg = sense, n.kwargs...)

        pred = sol.u[1]::TrackedArray{Float32,2,CuArray{Float32,2}}
        z = pred[1:end-3, :]
        delta_logp = pred[end-2, :]
        λ₁ = pred[end-1, :]
        λ₂ = pred[end, :]
    else
        _z = TrackedArray(CUDA.zeros(Float32, 1, size(x, 2)))

        prob = ODEProblem{false}(ffjord_, vcat(x, _z), tspan, p)
        sol = solve(prob, n.args...; sensealg = sense, n.kwargs...)

        pred = sol.u[1]::TrackedArray{Float32,2,CuArray{Float32,2}}
        z = pred[1:end-1, :]
        delta_logp = pred[end, :]
        λ₁ = _z[1, :]
        λ₂ = _z[1, :]
    end

    nfe = sol.destats.nf::Int
    logpz = reshape(sum(-(log(eltype(z)(2π)) .+ CUDA.pow.(z, 2)) ./ 2, dims = 1), :)
    logpx = logpz .- delta_logp

    return logpx, λ₁, λ₂, nfe, nothing
end

@fastmath function (n::TrackedFFJORD{true,M})(
    x,
    p = n.p,
    e = CUDA.randn(size(x)...);
    regularize = false,
) where {M}
    tspan = _convert_tspan(n.tspan, p)
    sense = SensitivityADPassThrough()
    sv = SavedValues(eltype(tspan), eltype(p))
    svcb = SavingCallback((u, t, integrator) -> integrator.EEst * integrator.dt, sv)
    ffjord_ = (u, p, t) -> _ffjord(u, p, t, n.re, e, false, M)
    _z = TrackedArray(CUDA.zeros(Float32, 1, size(x, 2)))

    prob = ODEProblem{false}(ffjord_, vcat(x, _z), tspan, p)
    sol = solve(prob, n.args...; sensealg = sense, callback = svcb, n.kwargs...)
    pred = sol.u[1]::TrackedArray{Float32,2,CuArray{Float32,2}}
    z = pred[1:end-1, :]
    delta_logp = pred[end, :]

    logpz = reshape(sum(-(log(eltype(z)(2π)) .+ CUDA.pow.(z, 2)) ./ 2, dims = 1), :)
    logpx = logpz .- delta_logp

    nfe = sol.destats.nf::Int

    return logpx, _z, _z, nfe, sv
end

function jacobian_fn(f, x::AbstractMatrix, t)
    y, back = Tracker.forward(f, x, t)
    z = similar(y)
    fill!(z, 0)
    vec = similar(x, size(x, 1), size(x, 1), size(x, 2))
    for i = 1:size(y, 1)
        z[i, :] .+= 1
        vec[i, :, :] = data(back(z)[1])
    end
    return vec, y
end

_trace_batched(x::AbstractArray{T,3}) where {T} =
    reshape([tr(x[:, :, i]) for i = 1:size(x, 3)], 1, size(x, 3))

function _deterministic_ffjord(u, p, t, re, M)
    m = re(p)::M
    z = u[1:end-1, :] |> untrack
    vec, mz = jacobian_fn(m, z, t)
    trace_jac = _trace_batched(vec)
    return vcat(mz, -trace_jac)
end

function sample(n::TrackedFFJORD{B,M}, p = n.p; nsamples::Int = 1) where {B,M}
    z_samples = CUDA.randn(size(n.re(p)[1].W, 2) - 1, nsamples)
    ffjord_ = (u, p, t) -> _deterministic_ffjord(u, p, t, n.re, M)
    _z = TrackedArray(CUDA.zeros(Float32, 1, nsamples))
    prob = ODEProblem{false}(ffjord_, vcat(z_samples, _z), [n.tspan[2], n.tspan[1]], p)
    x_gen = solve(prob, n.args...; sensealg = SensitivityADPassThrough(), n.kwargs...)
    return x_gen.u[1][1:end-1, :]
end
