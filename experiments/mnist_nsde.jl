#--------------------------------------
## LOAD PACKAGES
using RegNeuralDE, StochasticDiffEq, Flux, DiffEqFlux, Tracker
using YAML, Dates, BSON, Random, Statistics, Printf
using CUDA
using RegNeuralDE: accuracy
using Flux: @functor
using Tracker: TrackedReal
import Base.show

# This code will not work on GPU. Please trick your system to not detect GPUs
# using `CUDA_VISIBLE_DEVICES=""`. Will get this fixed soon
CUDA.allowscalar(false)
#--------------------------------------

#--------------------------------------
## CONFIGURATION
config_file = joinpath(pwd(), "experiments", "configs", "mnist_nsde.yml")
config = YAML.load_file(config_file)

Random.seed!(config["seed"])

hparams = config["hyperparameters"]
BATCH_SIZE = hparams["batch_size"]
EPOCHS = hparams["epochs"]
REGULARIZE = hparams["regularize"]
REG_TYPE = hparams["type"]
identifier =
    REGULARIZE ? "$(string(now()))_$(REGULARIZE)_$(REG_TYPE)" : "$(string(now()))_vanilla"
EXPERIMENT_LOGDIR = joinpath(pwd(), "results", "mnist_nsde", identifier)
MODEL_WEIGHTS = joinpath(EXPERIMENT_LOGDIR, "weights.bson")
FILENAME = joinpath(EXPERIMENT_LOGDIR, "results.yml")

# Create a directory to store the results
isdir(EXPERIMENT_LOGDIR) || mkpath(EXPERIMENT_LOGDIR)
cp(config_file, joinpath(EXPERIMENT_LOGDIR, "config.yml"))
#--------------------------------------

#--------------------------------------
## SETUP THE MODELS + DATASET + TRAINING UTILS
# Get the dataset
train_dataloader, test_dataloader = load_mnist(BATCH_SIZE, x -> flatten(Float32.(x)))

agg = mean
if REG_TYPE == "error_est"
    λ₀ = 1.0f1
    λ₁ = 1.0f1
    save_func(u, t, integrator) = integrator.EEst * integrator.dt
    solver = SOSRI()
    global agg = mean
elseif REG_TYPE == "stiff_est"
    λ₀ = 0.1f0
    λ₁ = 0.1f0
    const stability_size =
        Tracker.TrackedReal(1 / Float32(StochasticDiffEq.alg_stability_size(SOSRI2())))
    function save_func(u, t, integrator)
        stiff_est = abs(integrator.eigen_est)
        return stability_size * ((iszero(stiff_est) || isnan(stiff_est)) ? 0 : stiff_est)
    end
    global agg = mean
    solver = AutoSOSRI2(SOSRI2())
else
    global agg = mean
    solver = SOSRI()
end
k = log(λ₀ / λ₁) / EPOCHS
# Exponential Decay
λ_func(t) = λ₀ * exp(-k * t)

nsde = ClassifierNSDE(
    Dense(784, 32) |> track,
    TrackedNeuralDSDE(
        Chain(Dense(32, 64, tanh), Dense(64, 32)) |> track,
        Dense(32, 32) |> track,
        [0.0f0, 1.0f0],
        REGULARIZE,
        solver,
        save_everystep = false,
        reltol = 1.4f-1,
        abstol = 1.4f-1,
        save_start = false,
    ),
    Dense(32, 10) |> track,
)
ps = Flux.trainable(nsde)

opt = Flux.Optimise.Optimiser(InvDecay(1.0e-5), ADAM(0.01))

function loss_function(
    x,
    y,
    model,
    p1,
    p2,
    p3;
    trajectories = 1,
    λ = 1.0f2,
    notrack = false,
)
    pred, _, _, sv = model(x, p1, p2, p3; trajectories = trajectories, func = save_func)
    cross_entropy = Flux.Losses.logitcrossentropy(pred, y)
    reg = REGULARIZE ? λ * agg(sv.saveval) : zero(eltype(pred))
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
                "Regularization" => reg_un,
            ),
        )
    end
    return total_loss
end
#--------------------------------------

#--------------------------------------
## LOGGING UTILITIES
nfe1_counts = Vector{Float64}(undef, EPOCHS + 1)
nfe2_counts = Vector{Float64}(undef, EPOCHS + 1)
train_accuracies = Vector{Float64}(undef, EPOCHS + 1)
test_accuracies = Vector{Float64}(undef, EPOCHS + 1)
train_runtimes = Vector{Float64}(undef, EPOCHS + 1)
inference_runtimes = Vector{Float64}(undef, EPOCHS + 1)
train_runtimes[1] = 0

logger = table_logger(
    [
        "Epoch Number",
        "NFE1 Count",
        "NFE2 Count",
        "Train Accuracy",
        "Test Accuracy",
        "Train Runtime",
        "Inference Runtime",
    ],
    ["Total Loss", "Cross Entropy Loss", "Regularization"],
)
#--------------------------------------

#--------------------------------------
## RECORD DETAILS BEFORE TRAINING STARTS
dummy_data = train_dataloader.data[1][:, 1:BATCH_SIZE]
stime = time()
_, _nfe1, _nfe2, _ = nsde(dummy_data; func = save_func)
inference_runtimes[1] = time() - stime
train_runtimes[1] = 0.0
nfe1_counts[1] = _nfe1
nfe2_counts[1] = _nfe2
train_accuracies[1] = accuracy(nsde, train_dataloader; no_gpu = true, trajectories = 10)
test_accuracies[1] = accuracy(nsde, test_dataloader; no_gpu = true, trajectories = 10)

logger(
    false,
    Dict(),
    0.0,
    nfe1_counts[1],
    nfe2_counts[1],
    train_accuracies[1],
    test_accuracies[1],
    train_runtimes[1],
    inference_runtimes[1],
)
#--------------------------------------

#--------------------------------------
## WARMSTART THE GRADIENT COMPUTATION
y_ = zeros(Float32, 10, 1)
y_[1, :] .= 1.0
_ = Tracker.gradient(
    (p1, p2, p3) ->
        loss_function(rand(Float32, 784, 1), y_, nsde, p1, p2, p3; notrack = true),
    ps...,
)
#--------------------------------------

#--------------------------------------
## TRAINING
for epoch = 1:EPOCHS
    λ = λ_func(epoch - 1)
    timing = 0

    for (i, (x_, y_)) in enumerate(train_dataloader)
        x = x_
        y = y_

        start_time = time()
        gs = Tracker.gradient(
            (p1, p2, p3) -> loss_function(x, y, nsde, p1, p2, p3; λ = λ),
            ps...,
        )
        update_parameters!(ps, gs, opt)
        timing += time() - start_time

        x = y = nothing
        GC.gc(true)
    end

    # Record the time per epoch
    train_runtimes[epoch+1] = timing

    # Record the NFE count
    start_time = time()
    _, nfe1, nfe2, _ = nsde(dummy_data)
    inference_runtimes[epoch+1] = time() - start_time
    nfe1_counts[epoch+1] = nfe1
    nfe2_counts[epoch+1] = nfe2

    # Test and Train Accuracy
    train_accuracies[epoch+1] =
        accuracy(nsde, train_dataloader; no_gpu = true, trajectories = 10)
    test_accuracies[epoch+1] =
        accuracy(nsde, test_dataloader; no_gpu = true, trajectories = 10)

    logger(
        false,
        Dict(),
        epoch,
        nfe1_counts[epoch+1],
        nfe2_counts[epoch+1],
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
    :nfe1_counts => nfe1_counts,
    :nfe2_counts => nfe2_counts,
    :train_accuracies => train_accuracies,
    :test_accuracies => test_accuracies,
    :train_runtimes => train_runtimes,
    :inference_runtimes => inference_runtimes,
)

weights = Flux.params(nsde) .|> cpu .|> untrack
BSON.@save MODEL_WEIGHTS weights

YAML.write_file(FILENAME, results)
#--------------------------------------
