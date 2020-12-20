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
    xarr = x.u[end]
    return reshape(xarr, size(xarr)[1:2])
end

# function diffeqsol_to_trackedarray(x)
#     return hcat(map(_x -> reshape(_x, size(_x, 1), 1, size(_x, 2)), x.u)...) 
# end

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

# Pretty Logger
function table_logger(header::Vector{String}, record::Vector{String} = [])
    n = length(header) + length(record)
    ind_lens = vcat(length.(header), length.(record))
    span = sum(ind_lens .+ 3) + 1
    println("=" ^ span)
    for h in vcat(header, record)
        print("| $h ")
    end
    println("|")
    println("=" ^ span)

    avg_meters = Dict{String, AverageMeter}(
        rec => AverageMeter() for rec in record
    )

    patterns = ["%$l.4f" for l in ind_lens]
    fmtrfuncs = generate_formatter.(patterns)
    function internal_logger(last::Bool, records::Dict, args::Vararg)
        if length(records) > 0
            for (rec, val) in records
                update!(avg_meters[rec], val)
            end
            return
        end
        if last
            println("=" ^ span)
            return
        end
        for h in [fmtrfunc(arg) for (fmtrfunc, arg) in zip(
            fmtrfuncs, vcat([args...], [avg_meters[rec]() for rec in record])
        )]
            print("| $h ")
        end
        println("|")
    end
end
