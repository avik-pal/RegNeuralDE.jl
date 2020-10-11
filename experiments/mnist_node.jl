using RegNeuralODE, OrdinaryDiffEq, Flux, DiffEqFlux, Tracker, Random, Statistics
using ProgressLogging, YAML, Dates, BSON
using CUDA
using RegNeuralODE: accuracy
using Flux.Optimise: update!
using Flux: @functor, glorot_uniform, logitcrossentropy
using Tracker: TrackedReal, data
import Base.show

## Training Parameters
config_file = joinpath(pwd(), "experiments", "configs", "mnist_node.yml")
config = YAML.load_file(config_file)

Random.seed!(config["seed"])

hparams = config["hyperparameters"]
BATCH_SIZE = hparams["batch_size"]
REGULARIZE = hparams["regularize"]
HIDDEN_DIMS = hparams["hidden_dims"]
INPUT_DIMS = hparams["input_dims"]
LR = hparams["lr"]
EPOCHS = hparams["epochs"]
EXPERIMENT_LOGDIR = joinpath(config["log_dir"], string(now()))
MODEL_WEIGHTS = joinpath(EXPERIMENT_LOGDIR, "weights.bson")
FILENAME = joinpath(EXPERIMENT_LOGDIR, "results.yml")

# Create a directory to store the results
isdir(EXPERIMENT_LOGDIR) || mkpath(EXPERIMENT_LOGDIR)
cp(config_file, joinpath(EXPERIMENT_LOGDIR, "config.yml"))

# This model is same as the ones in https://github.com/jacobjinkelly/easy-neural-ode/blob/master/mnist.py

# The dynamics of the Neural ODE are time dependent
struct MLPDynamics{W1, B1, W2, B2}
    weight_1::W1
    bias_1::B1
    weight_2::W2
    bias_2::B2
end

Flux.@functor MLPDynamics

function MLPDynamics(in::Integer, hidden::Integer)
    return MLPDynamics(glorot_uniform(hidden, in + 1), zeros(Float32, hidden),
                       glorot_uniform(in, hidden + 1), zeros(Float32, in))
end

function (mlp::MLPDynamics)(x::AbstractMatrix, t::TrackedReal)
    _t = Tracker.collect(ones(eltype(x), 1, size(x, 2))) .* t
    z = mlp.weight_1 * vcat(σ.(x), _t) .+ mlp.bias_1
    return mlp.weight_2 * vcat(σ.(z), _t) .+ mlp.bias_2
end

Base.show(io::IO, mlp::MLPDynamics) =
    print(io, "Time Dependent MLP Dynamics: $(size(mlp.weight_1, 2) - 1) -> $(size(mlp.weight_2, 2) - 1) -> $(size(mlp.weight_2, 1))")


# Get the dataset
train_dataloader, test_dataloader = load_mnist(BATCH_SIZE, x -> cpu(track(x)))

# Define the models
mlp_dynamics = MLPDynamics(64, 128)

node = ClassifierNODE(
    Chain(x -> reshape(x, 784, :), Linear(784, 256, relu), Linear(256, 64)) |> track,
    TrackedNeuralODE(mlp_dynamics |> track, [0.f0, 1.f0], true,
                     REGULARIZE, Tsit5(), save_everystep = false,
                     reltol = 6f-5, abstol = 6f-5, save_start = false),
    Chain(Linear(64, 10)) |> track
)

opt = ADAMW(LR, (0.9, 0.99), 1e-5)

function loss_function(x, y, model, p1, p2, p3; λ = 1.0f2)
    pred, sol, sv = model(x, p1, p2, p3)
    return logitcrossentropy(pred, y) +(
        REGULARIZE ? λ * mean(sv.saveval) : zero(eltype(pred))
    )
end

nfe_counts = Vector{Float64}(undef, EPOCHS + 1)
train_accuracies = Vector{Float64}(undef, EPOCHS + 1)
test_accuracies = Vector{Float64}(undef, EPOCHS + 1)
train_runtimes = Vector{Float64}(undef, EPOCHS + 1)  # The first value is a dummy value
inference_runtimes = Vector{Float64}(undef, EPOCHS + 1)

train_runtimes[1] = 0

ps = Flux.trainable(node)

dummy_data = rand(Float32, 28, 28, 1, 1) |> track
start_time = time()
_, sol, _ = node(dummy_data)
inference_runtimes[1] = time() - start_time
train_runtimes[1] = 0.0
nfe_counts[1] = sol.destats.nf
train_accuracies[1] = accuracy(node, train_dataloader)
test_accuracies[1] = accuracy(node, test_dataloader)
@info (train_runtimes[1], inference_runtimes[1], nfe_counts[1], train_accuracies[1], test_accuracies[1])

@progress for epoch in 1:EPOCHS
    start_time = time()

    @progress for (i, (x, y)) in enumerate(train_dataloader)
        gs = Tracker.gradient((p1, p2, p3) -> loss_function(x, y, node, p1, p2, p3), ps...)
        for (p, g) in zip(ps, gs)
            length(p) == 0 && continue
            update!(opt, data(p), data(g))
        end
    end
    # Record the time per epoch
    train_runtimes[epoch + 1] = time() - start_time

    # Record the NFE count
    start_time = time()
    _, sol, _ = node(dummy_data)
    inference_runtimes[epoch + 1] = time() - start_time
    nfe_counts[epoch + 1] = sol.destats.nf

    # Test and Train Accuracy
    train_accuracies[epoch + 1] = accuracy(node, train_dataloader)
    test_accuracies[epoch + 1] = accuracy(node, test_dataloader)

    @info (train_runtimes[epoch + 1], inference_runtimes[epoch + 1], nfe_counts[epoch + 1], train_accuracies[epoch + 1], test_accuracies[epoch + 1])
end

results = Dict(
    :nfe_counts => nfe_counts,
    :train_accuracies => train_accuracies,
    :test_accuracies => test_accuracies,
    :train_runtimes => train_runtimes,
    :inference_runtimes => inference_runtimes
)

BSON.@save MODEL_WEIGHTS Dict(
    :p1 => node.p1,
    :p2 => node.p2,
    :p3 => node.p3
)

YAML.write_file(FILENAME, results)
