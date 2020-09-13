# Metrics
classify(x) = argmax.(eachcol(x))

function accuracy(model, data; batches = length(data))
    total_correct = 0
    total = 0
    for (i, (x, y)) in enumerate(collect(data))
        i >= batches && break
        target_class = classify(cpu(y))
        predicted_class = _get_predicted_class(model, x)
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

_get_predicted_class(model::ClassifierNODE, x) = classify(cpu(model(x)))
    
_get_predicted_class(model::ClassifierNODE{T}, x) where T<:NFECounterCallbackNeuralODE =
    classify(cpu(model(x)[1]))


# Loss Functions
get_loss_function(model::ClassifierNODE) =
    (x, y, model, ps...) -> Flux.logitcrossentropy(model(x, ps...), y)

function get_loss_function(model::ClassifierNODE{T}; λ = 1.0f2) where T<:NFECounterCallbackNeuralODE
    function regularized_logitcrossentropy_loss(x, y, model, ps...)
        pred, sv = model(x, ps...)
        return Flux.logitcrossentropy(pred, y) + λ * mean(sv.saveval)
    end
    return regularized_logitcrossentropy_loss
end

function get_loss_function(model::ExtrapolationLatentODE)
    function loss_extrapolating_latent_ode(x::AbstractArray{T},
                                           model, ps...) where T
        pred, qz0_μ, qz0_logvar = model(x, ps...)
        logpx = reshape(sum(log_normal_pdf(x, pred, T(2 * log(0.3))),
                            dims=(1, 2)), :)
        analytic_kl = reshape(sum(normal_kl(qz0_μ, qz0_logvar, zeros(T, size(qz0_μ)),
                                            zeros(T, size(qz0_logvar))),
                                  dims = 1), :)

        return mean(analytic_kl .- logpx)
    end
    return loss_extrapolating_latent_ode
end

function get_loss_function(model::ExtrapolationLatentODE{M}; λ = 1.0f2) where M<:NFECounterCallbackNeuralODE
    function regularized_loss_extrapolating_latent_ode(x::AbstractArray{T},
                                                       model, ps...) where T
        pred, qz0_μ, qz0_logvar, sv = model(x, ps...)
        logpx = reshape(sum(log_normal_pdf(x, pred, T(2 * log(0.3))),
                            dims=(1, 2)), :)
        analytic_kl = reshape(sum(normal_kl(qz0_μ, qz0_logvar, zeros(T, size(qz0_μ)),
                                            zeros(T, size(qz0_logvar))),
                                  dims = 1), :)

        return mean(analytic_kl .- logpx) + λ * mean(sv.saveval)
    end
    return regularized_loss_extrapolating_latent_ode
end

log_normal_pdf(x::AbstractArray{T}, mean, logvar) where T =
    -(log(T(2π)) .+ logvar .+ ((x .- mean) .^ 2) ./ exp.(logvar)) ./ 2

function normal_kl(μ₁::AbstractArray{T1}, logvar₁::AbstractArray{T1},
                   μ₂::AbstractArray{T2}, logvar₂::AbstractArray{T2}) where {T1, T2}
    v₁ = exp.(logvar₁)
    v₂ = exp.(logvar₂)

    lstd₁ = logvar₁ .* T1(0.5)
    lstd₂ = logvar₂ .* T2(0.5)

    return lstd₂ .- lstd₁ .+ ((v₁ .+ (μ₁ .- μ₂) .^ 2) ./ (2 .* v₂)) .- T2(0.5)
end

# DiffEqSolution to Array
function diffeqsol_to_array(x)
    xarr = cpu(x)
    return reshape(xarr, size(xarr)[1:2])
end

function diffeqsol_to_cuarray(x)
    xarr = gpu(x)
    return reshape(xarr, size(xarr)[1:2])
end

function diffeqsol_to_trackedarray(x)
    xarr = x.u[1]
    return reshape(xarr, size(xarr)[1:2])
end

_convert_tspan(tspan, p) = eltype(p).(tspan)

_convert_tspan(tspan, p::TrackedArray) = Tracker.collect(eltype(p).(tspan))

norm_batched(x::AbstractArray) = sqrt.(sum(x .^ 2, dims = 1))

# Running Average Meter
mutable struct AverageMeter{T}
    last_value::T
    sum::T
    count::Int
    
    AverageMeter(T = Float32) = new{T}(T(0), T(0), 0)
end

function reset!(am::AverageMeter{T}) where T
    am.last_value = T(0)
    am.sum = T(0)
    am.count = 0
    return am
end

function update!(am::AverageMeter{T}, val::T) where T
    am.last_value = val
    am.sum += val
    am.count += 1
    return am.sum / am.count
end

(am::AverageMeter)() = am.sum / am.count