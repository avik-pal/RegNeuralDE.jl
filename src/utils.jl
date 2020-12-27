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

function diffeqsol_to_3dtrackedarray(x)
    return hcat(map(_x -> reshape(_x, size(_x, 1), 1, size(_x, 2)), x.u)...)
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

function reset!(am::AverageMeter{T}) where {T}
    am.last_value = T(0)
    am.sum = T(0)
    am.count = 0
    return am
end

function update!(am::AverageMeter{T}, val::T) where {T}
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
    println("="^span)
    for h in vcat(header, record)
        print("| $h ")
    end
    println("|")
    println("="^span)

    avg_meters = Dict{String,AverageMeter}(rec => AverageMeter() for rec in record)

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
            println("="^span)
            return
        end
        for h in [
            fmtrfunc(arg)
            for
            (fmtrfunc, arg) in
            zip(fmtrfuncs, vcat([args...], [avg_meters[rec]() for rec in record]))
        ]
            print("| $h ")
        end
        println("|")
    end
end

# MvNormal for FFJORD
struct BatchedMultiVariateNormal{M, N, O, P, Q}
    μ::M
    cov::N
    inv_cov::N
    det_cov::O
    lt_decom::P
    k::Q
end

Flux.@functor BatchedMultiVariateNormal

function BatchedMultiVariateNormal(μ::AbstractVector, cov::AbstractMatrix)
    inv_cov = inv(cov)
    det_cov = det(cov)
    lt_decom = Array(cholesky(cov))
    return BatchedMultiVariateNormal(reshape(μ, :, 1), cov, inv_cov, det_cov, lt_decom, length(μ))
end

function (mvnorm::BatchedMultiVariateNormal)(x::AbstractMatrix{T}) where T
    denom = sqrt((T(2π) ^ mvnorm.k) * T(mvnorm.det_cov))
    diff = x .- mvnorm.μ
    return exp.(diag(-transpose(diff) * (mvnorm.inv_cov * diff) / 2)) / denom
end

function sample(mvnorm::BatchedMultiVariateNormal{CuArray{T, 2}}, nsamples::Int) where T
    samples = CUDA.randn(T, mvnorm.k, nsamples) :: CuArray{T, 2}
    return mvnorm.μ .+ mvnorm.lt_decom * samples
end

function sample(mvnorm::BatchedMultiVariateNormal{TrackedArray{T, 2, CuArray{T, 2}}}, nsamples::Int) where T
    samples = CUDA.randn(T, mvnorm.k, nsamples) :: CuArray{T, 2}
    return mvnorm.μ .+ mvnorm.lt_decom * samples
end

function sample(mvnorm::BatchedMultiVariateNormal{Array{T, 2}}, nsamples::Int) where T
    samples = randn(T, mvnorm.k, nsamples)
    return mvnorm.μ .+ mvnorm.lt_decom * samples
end

function sample(mvnorm::BatchedMultiVariateNormal{TrackedArray{T, 2, Array{T, 2}}}, nsamples::Int) where T
    samples = randn(T, mvnorm.k, nsamples)
    return mvnorm.μ .+ mvnorm.lt_decom * samples
end