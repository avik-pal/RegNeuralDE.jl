# Metrics
classify(x) = argmax.(eachcol(x))

function accuracy(model, data; batches = length(data), no_gpu::Bool = false, kwargs...)
    total_correct = 0
    total = 0
    for (i, (x_, y)) in enumerate(collect(data))
        i > batches && break
        x = no_gpu ? x_ : (x_ |> gpu |> track)
        target_class = classify(cpu(y))
        predicted_class = classify(cpu(model(x; kwargs...)[1] |> untrack))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
        x_ = nothing
        GC.gc(true)
    end
    return total_correct * 100 / total
end

function loglikelihood(model, data; batches = length(data))
    total_loglikelihood = 0.0f0
    total = 0
    for (i, x_) in enumerate(data)
        i > batches && break
        x = x_ |> gpu
        res = model(x |> track)
        total_loglikelihood += sum(res[1])
        total += size(x, 2)
        x_ = nothing
        GC.gc(true)
    end
    return total_loglikelihood / total
end
