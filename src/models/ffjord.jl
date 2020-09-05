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
            # model should contain Linear layers (not Dense)
            size_input = size(model[1].weight)[2]
            T = eltype(model[1].weight)
            basedist = MvNormal(zeros(T, size_input),
                                I + zeros(T, size_input, size_input))
        end
        new{typeof(model), typeof(p), typeof(re), typeof(basedist),
            typeof(tspan), typeof(args), typeof(kwargs)}(
            model, p, re, basedist, tspan, args, kwargs, [0])
    end
end

function _norm_batched(x::AbstractMatrix)
    res = similar(x, 1, size(x, 2))
    for i in 1:size(x, 2)
        res[1, i] = norm(@view x[:, i])
    end
    return res
end

function ffjord(u, p, t, re, e, regularize)
    m = re(p)
    if regularize
        z = @view u[1:end - 3, :]
        mz, back = Zygote.pullback(m, z)
        eJ = back(e)[1]
        trace_jac = sum(eJ .* e, dims = 1)
        return cat(mz, -trace_jac, sum(abs2, mz, dims=1),
                   _norm_batched(eJ) .^ 2, dims=1)
    else
        z = @view u[1:end - 1, :]
        mz, back = Zygote.pullback(m, z)
        eJ = back(e)[1]
        trace_jac = sum(eJ .* e, dims = 1)
        res = cat(mz, -trace_jac, dims=1)
        return res
    end
end

function (n::NFECounterFFJORD)(x, p = n.p, regularize = false)
    e = randn(eltype(x), size(x))
    pz = n.basedist
    sense = InterpolatingAdjoint(autojacvec = false)
    ffjord_ = (u, p, t) -> begin
        Zygote.@ignore n.nfe[] += 1
        return ffjord(u, p, t, n.re, e, regularize)
    end
    if regularize
        _z = zeros(eltype(x), 3, size(x, 2))
        prob = ODEProblem{false}(ffjord_, vcat(x, _z), n.tspan, p)
        pred = solve(prob, n.args...; sensealg = sense, n.kwargs...)[:, :, end]
        z = @view pred[1:end - 3, :]
        delta_logp = reshape(pred[end - 2, :], 1, size(pred, 2))
        λ₁ = @view pred[end - 1, :]
        λ₂ = @view pred[end, :]
    else
        _z = zeros(eltype(x), 1, size(x, 2))
        prob = ODEProblem{false}(ffjord_, vcat(x, _z), n.tspan, p)
        pred = solve(prob, n.args...; sensealg = sense, n.kwargs...)[:, :, end]
        z = @view pred[1:end - 1, :]
        delta_logp = reshape(pred[end, :], 1, size(pred, 2))
        λ₁ = λ₂ = @view _z[1, :]
    end

    # logpdf promotes the type to Float64 by default
    logpz = reshape(logpdf(pz, z), 1, size(x, 2))
    logpx = logpz .- delta_logp

    return logpx, λ₁, λ₂
end