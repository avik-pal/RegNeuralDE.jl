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