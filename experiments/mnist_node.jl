#--------------------------------------
## LOAD PACKAGES
using RegNeuralODE, OrdinaryDiffEq, Flux, DiffEqFlux, Tracker
using YAML, Dates, BSON, Random, Statistics, Printf
using CUDA
using RegNeuralODE: accuracy
using Flux.Optimise: update!
using Flux: @functor, glorot_uniform, logitcrossentropy
using Tracker: TrackedReal, data
import Base.show

CUDA.allowscalar(false)
#--------------------------------------

#--------------------------------------
## CONFIGURATION
config_file = joinpath(pwd(), "experiments", "configs", "mnist_node.yml")
config = YAML.load_file(config_file)

Random.seed!(config["seed"])

hparams = config["hyperparameters"]
BATCH_SIZE = hparams["batch_size"]
REGULARIZE = hparams["regularize"]
EPOCHS = hparams["epochs"]
EXPERIMENT_LOGDIR = joinpath(pwd(), "results", "mnist_node", "$(string(now()))_$REGULARIZE")
MODEL_WEIGHTS = joinpath(EXPERIMENT_LOGDIR, "weights.bson")
FILENAME = joinpath(EXPERIMENT_LOGDIR, "results.yml")

# Create a directory to store the results
isdir(EXPERIMENT_LOGDIR) || mkpath(EXPERIMENT_LOGDIR)
cp(config_file, joinpath(EXPERIMENT_LOGDIR, "config.yml"))
#--------------------------------------

#--------------------------------------
## NEURAL NETWORK
# The dynamics of the Neural ODE are time dependent
struct MLPDynamics{D1,D2}
    dense_1::D1
    dense_2::D2
end

@functor MLPDynamics

MLPDynamics(in::Integer, hidden::Integer) =
    MLPDynamics(Dense(in + 1, hidden, CUDA.tanh), Dense(hidden + 1, in, CUDA.tanh))

function (mlp::MLPDynamics)(x::AbstractMatrix, t::TrackedReal)
    _t = CUDA.ones(Float32, 1, size(x, 2)) .* t
    return mlp.dense_2(vcat(mlp.dense_1(vcat(x, _t)), _t))
end
#--------------------------------------

#--------------------------------------
## SETUP THE MODELS + DATASET + TRAINING UTILS
# Get the dataset
train_dataloader, test_dataloader = load_mnist(BATCH_SIZE, x -> gpu(track(x)))

# Define the models
mlp_dynamics = MLPDynamics(784, 100)

node = ClassifierNODE(
    Chain(x -> reshape(x, 784, :)) |> track |> gpu,
    TrackedNeuralODE(
        mlp_dynamics |> track |> gpu,
        [0.0f0, 1.0f0],
        true,
        REGULARIZE,
        Tsit5(),
        save_everystep = false,
        reltol = 1.4f-8,
        abstol = 1.4f-8,
        save_start = false,
    ),
    Dense(784, 10) |> track |> gpu,
)
ps = Flux.trainable(node)

opt = Flux.Optimise.Optimiser(InvDecay(1.0e-5), Momentum(0.1, 0.9))

# Anneal the regularization so that it doesn't overpower the
# the main objective
λ₀ = 1.0f2
λ₁ = 1.0f1
k = log(λ₀ / λ₁) / EPOCHS
# Exponential Decay
λ_func(t) = λ₀ * exp(-k * t)

function loss_function(x, y, model, p1, p2, p3; λ = 1.0f2, notrack = false)
    pred, _, sv = model(x, p1, p2, p3)
    cross_entropy = Flux.Losses.logitcrossentropy(pred, y)
    reg = REGULARIZE ? λ * mean(sv.saveval) : zero(eltype(pred))
    total_loss = cross_entropy + reg
    if !notrack
        ce_un = cross_entropy |> untrack
        reg_un = reg |> untrack
        total_loss_un = total_loss |> untrack
        logger(
            false,
            Dict(
                "Total Loss" => total_loss_un,
                "Cross Entropy Loss" => ce_un,
                "Our Regularization" => reg_un,
            ),
        )
    end
    return total_loss
end
#--------------------------------------

#--------------------------------------
## LOGGING UTILITIES
nfe_counts = Vector{Float64}(undef, EPOCHS + 1)
train_accuracies = Vector{Float64}(undef, EPOCHS + 1)
test_accuracies = Vector{Float64}(undef, EPOCHS + 1)
train_runtimes = Vector{Float64}(undef, EPOCHS + 1)  # The first value is a dummy value
inference_runtimes = Vector{Float64}(undef, EPOCHS + 1)
train_runtimes[1] = 0

logger = table_logger(
    [
        "Epoch Number",
        "NFE Count",
        "Train Accuracy",
        "Test Accuracy",
        "Train Runtime",
        "Inference Runtime",
    ],
    ["Total Loss", "Cross Entropy Loss", "Our Regularization"],
)
#--------------------------------------

#--------------------------------------
## RECORD DETAILS BEFORE TRAINING STARTS
dummy_data = rand(Float32, 28, 28, 1, BATCH_SIZE) |> track |> gpu
stime = time()
_, _nfe, _ = node(dummy_data)
inference_runtimes[1] = time() - stime
train_runtimes[1] = 0.0
nfe_counts[1] = _nfe
train_accuracies[1] = accuracy(node, train_dataloader)
test_accuracies[1] = accuracy(node, test_dataloader)

logger(
    false,
    Dict(),
    0.0,
    nfe_counts[1],
    train_accuracies[1],
    test_accuracies[1],
    train_runtimes[1],
    inference_runtimes[1],
)
#--------------------------------------

#--------------------------------------
## WARMSTART THE GRADIENT COMPUTATION
y_ = zeros(Float32, 10, 1)
y_[1, 1] = 1.0
_ = Tracker.gradient(
    (p1, p2, p3) -> loss_function(
        rand(Float32, 28, 28, 1, 1) |> gpu |> track,
        y_ |> gpu |> track,
        node,
        p1,
        p2,
        p3;
        notrack = true,
    ),
    ps...,
)
#--------------------------------------

#--------------------------------------
## TRAINING
for epoch = 1:EPOCHS
    λ = λ_func(epoch - 1)

    start_time = time()

    for (i, (x, y)) in enumerate(train_dataloader)
        gs = Tracker.gradient(
            (p1, p2, p3) -> loss_function(x, y, node, p1, p2, p3; λ = λ),
            ps...,
        )
        for (p, g) in zip(ps, gs)
            length(p) == 0 && continue
            update!(opt, data(p), data(g))
        end
    end

    # Record the time per epoch
    train_runtimes[epoch+1] = time() - start_time

    # Record the NFE count
    start_time = time()
    _, nfe, _ = node(dummy_data)
    inference_runtimes[epoch+1] = time() - start_time
    nfe_counts[epoch+1] = nfe

    # Test and Train Accuracy
    train_accuracies[epoch+1] = accuracy(node, train_dataloader)
    test_accuracies[epoch+1] = accuracy(node, test_dataloader)

    logger(
        false,
        Dict(),
        epoch,
        nfe_counts[epoch+1],
        train_accuracies[epoch+1],
        test_accuracies[epoch+1],
        train_runtimes[epoch+1],
        inference_runtimes[epoch+1],
    )
end
logger(true, Dict())
#--------------------------------------

#--------------------------------------
## STORE THE RESULTS
results = Dict(
    :nfe_counts => nfe_counts,
    :train_accuracies => train_accuracies,
    :test_accuracies => test_accuracies,
    :train_runtimes => train_runtimes,
    :inference_runtimes => inference_runtimes,
)

weights = Flux.params(node) .|> cpu .|> untrack
BSON.@save MODEL_WEIGHTS weights

YAML.write_file(FILENAME, results)
#--------------------------------------
