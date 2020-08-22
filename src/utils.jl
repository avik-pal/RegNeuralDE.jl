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

# DiffEqSolution to Array
function diffeqsol_to_array(x)
    xarr = cpu(x)
    return reshape(xarr, size(xarr)[1:2])
end

function diffeqsol_to_cuarray(x)
    xarr = gpu(x)
    return reshape(xarr, size(xarr)[1:2])
end