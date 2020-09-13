struct NFECounterFFJORD{M,P,RE,D,T,A,K} <: DiffEqFlux.CNFLayer
    model::M
    p::P
    re::RE
    basedist::D
    tspan::T
    args::A
    kwargs::K
    nfe::Vector{Int}

    function NFECounterFFJORD(model, tspan, args...; basedist = nothing, kwargs...)
        p, re = Flux.destructure(model)
        if basedist === nothing
            size_input = size(hasproperty(model[1], :weight) ? model[1].weight : model[1].W)[2]
            T = eltype(model[1].weight)
            basedist = MvNormal(zeros(Float32, size_input),
                                I + zeros(Float32, size_input, size_input))
        end
        new{typeof(model), typeof(p), typeof(re), typeof(basedist),
            typeof(tspan), typeof(args), typeof(kwargs)}(
            model, p, re, basedist, tspan, args, kwargs, [0])
    end
end

norm_batched(x::AbstractArray) = sqrt.(sum(x .^ 2, dims = 1))

function ffjord(u, p, t, re, e, regularize)
    m = re(p)
    if regularize
        z = u[1:end - 3, :]
        mz, back = Tracker.forward(m, z)
        eJ = back(e)[1]
        trace_jac = sum(eJ .* e, dims = 1)
        return Tracker.collect(cat(mz, -trace_jac, sum(abs2.(mz), dims = 1),
                                   norm_batched(eJ) .^ 2, dims = 1))
    else
        z = u[1:end - 1, :]
        mz, back = Tracker.forward(m, z)
        eJ = back(e)[1]
        trace_jac = sum(eJ .* e, dims = 1)
        return Tracker.collect(cat(mz, -trace_jac, dims = 1))
    end
end

function (n::NFECounterFFJORD)(x, p = n.p,
                               e = Tracker.collect(randn(eltype(x), size(x))),
                               regularize = false)
    pz = n.basedist
    sense = SensitivityADPassThrough()
    ffjord_ = (u, p, t) -> begin
        n.nfe[] += 1
        return ffjord(u, p, t, n.re, e, regularize)
    end
    if regularize
        _z = Tracker.collect(zeros(eltype(x), 3, size(x, 2)))
        prob = ODEProblem{false}(ffjord_, vcat(x, _z), n.tspan, p)
        pred = solve(prob, n.args...; sensealg = sense, n.kwargs...)[:, :, end]
        z = Tracker.collect(pred[1:end - 3, :])
        delta_logp = reshape(Tracker.collect(pred[end - 2, :]), 1, size(pred, 2))
        λ₁ = Tracker.collect(pred[end - 1, :])
        λ₂ = Tracker.collect(pred[end, :])
    else
        _z = Tracker.collect(zeros(eltype(x), 1, size(x, 2)))
        prob = ODEProblem{false}(ffjord_, vcat(x, _z), n.tspan, p)
        pred = solve(prob, n.args...; sensealg = sense, n.kwargs...)[:, :, end]
        z = Tracker.collect(pred[1:end - 1, :])
        delta_logp = reshape(Tracker.collect(pred[end, :]), 1, size(pred, 2))
        λ₁ = λ₂ = Tracker.collect(_z[1, :])
    end

    # logpdf promotes the type to Float64 by default
    # This function is type unstable when used with Tracker
    logpz = reshape(logpdf(pz, z), 1, size(x, 2))
    logpx = logpz .- delta_logp

    return logpx, λ₁, λ₂
end