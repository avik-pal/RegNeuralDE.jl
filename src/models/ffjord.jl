struct TrackedFFJORD{R,M,P,RE,D,T,A,K} <: DiffEqFlux.CNFLayer
    model::M
    p::P
    re::RE
    basedist::D
    tspan::T
    args::A
    kwargs::K

    function TrackedFFJORD(model, tspan, regularize, in_dims, args...; kwargs...)
        p, re = Flux.destructure(model)
        size_input = in_dims
        basedist = BatchedMultiVariateNormal(
            zeros(Float32, size_input),
            (I + zeros(Float32, size_input, size_input)),
        )
        basedist = basedist |> gpu
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

@fastmath function _ffjord(u, p, t, re, e, regularize, M)
    m = re(p)::M
    if regularize
        z = u[1:end-3, :]
        mz, back = Tracker.forward(z -> m(z, t), z)
        eJ = back(e)[1]
        trace_jac = sum(eJ .* e, dims = 1)
        return vcat(mz, -trace_jac, sum(abs2.(mz), dims = 1), norm_batched(eJ) .^ 2)
    else
        z = u[1:end-1, :]
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
    logpz = CUDA.log.(pz(z))
    logpx = logpz .- delta_logp

    return logpx, λ₁, λ₂, nfe, nothing
end

@fastmath function (n::TrackedFFJORD{true,M})(
    x,
    p = n.p,
    e = CUDA.randn(size(x)...);
    regularize = false,
) where {M}
    pz = n.basedist
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

    logpz = CUDA.log.(pz(z))
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
    z = u[1:end-1, :]
    vec, mz = jacobian_fn(m, z, t)
    trace_jac = _trace_batched(vec)
    return vcat(mz, -trace_jac)
end

function sample(n::TrackedFFJORD{B,M}, p = n.p; nsamples::Int = 1) where {B,M}
    pz = n.basedist
    z_samples = sample(pz, nsamples)
    ffjord_ = (u, p, t) -> _deterministic_ffjord(u, p, t, n.re, M)
    _z = TrackedArray(CUDA.zeros(Float32, 1, nsamples))
    prob = ODEProblem{false}(ffjord_, vcat(z_samples, _z), [n.tspan[2], n.tspan[1]], p)
    x_gen = solve(prob, n.args...; sensealg = SensitivityADPassThrough(), n.kwargs...)
    return x_gen.u[1][1:end-1, :]
end
