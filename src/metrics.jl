# Metrics
classify(x) = argmax.(eachcol(x))

function accuracy(model, data; batches = length(data))
    total_correct = 0
    total = 0
    for (i, (x, y)) in enumerate(collect(data))
        i >= batches && break
        target_class = classify(cpu(y))
        predicted_class = classify(cpu(model(x)[1]))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct * 100 / total
end

function loglikelihood(model, data; batches = length(data))
    total_loglikelihood = 0.0f0
    total = 0
    for (i, x) in enumerate(data)
        i >= batches && break
        res = model(x)
        total_loglikelihood += sum(res[1])
        total += size(x, 2)
    end
    return total_loglikelihood / total
end

# _get_predicted_class(model::ClassifierNODE, x) = classify(cpu(model(x)))
    
# _get_predicted_class(model::ClassifierNODE{T}, x) where T<:NFECounterCallbackNeuralODE =
    # classify(cpu(model(x)[1]))