#--------------------------------------
## LOAD PACKAGES
using RegNeuralODE, OrdinaryDiffEq, Flux, DiffEqFlux, Tracker, Random, Statistics
using YAML, Dates, BSON
using CUDA
using RegNeuralODE: loglikelihood
using Flux.Optimise: update!
using Flux: @functor
using Tracker: TrackedReal, data
import Base.show

CUDA.allowscalar(false)
#--------------------------------------

#--------------------------------------
## CONFIGURATION
## Training Parameters
config_file = joinpath(pwd(), "experiments", "configs", "ffjord_gaussian.yml")
config = YAML.load_file(config_file)

Random.seed!(config["seed"])

hparams = config["hyperparameters"]
const BATCH_SIZE = hparams["batch_size"]
const REGULARIZE = hparams["regularize"]
EPOCHS = hparams["epochs"]
EXPERIMENT_LOGDIR =
    joinpath(pwd(), "results", "ffjord_gaussian", "$(string(now()))_$REGULARIZE")
MODEL_WEIGHTS = joinpath(EXPERIMENT_LOGDIR, "weights.bson")
FILENAME = joinpath(EXPERIMENT_LOGDIR, "results.yml")

# Create a directory to store the results
isdir(EXPERIMENT_LOGDIR) || mkpath(EXPERIMENT_LOGDIR)
cp(config_file, joinpath(EXPERIMENT_LOGDIR, "config.yml"))
#--------------------------------------

#--------------------------------------
## SETUP THE MODELS + DATASET + TRAINING UTILS
# Get the dataset
train_dataloader, test_dataloader =
    load_gaussian_mixture(BATCH_SIZE, x -> gpu(x), ngaussians = 6, nsamples = 2048)

nn_dynamics =
    TDChain(Dense(3, 8, CUDA.tanh), Dense(9, 8, CUDA.tanh), Dense(9, 2)) |> gpu |> track

ffjord = TrackedFFJORD(
    nn_dynamics,
    [0.0f0, 1.0f0],
    REGULARIZE,
    Tsit5(),
    save_everystep = false,
    reltol = 1.4f-8,
    abstol = 1.4f-8,
    save_start = false,
)

ps = Flux.trainable(ffjord)

opt = Flux.Optimise.Optimiser(WeightDecay(1e-3), ADAM(4e-2))

# Anneal the regularization so that it doesn't overpower the
# the main objective
λ₀ = 5.0f3
λ₁ = 2.5f3
k = log(λ₀ / λ₁) / EPOCHS
# Exponential Decay
λ_func(t) = λ₀ * exp(-k * t)

function loss_function(x, model, p; λᵣ = 1.0f2, notrack = false)
    logpx, r1, r2, nfe, sv = model(x, p)
    neg_log_likelihood = -mean(logpx)
    reg = REGULARIZE ? λᵣ * mean(sv.saveval) : 0.0f0
    total_loss = neg_log_likelihood + reg
    if !notrack
        ll_un = neg_log_likelihood |> untrack
        reg_un = reg |> untrack
        total_loss_un = total_loss |> untrack
        logger(
            false,
            Dict(
                "Total Loss" => total_loss_un,
                "Negative Log Likelihood" => ll_un,
                "Regularization" => reg_un,
            ),
        )
    end
    return total_loss
end
#--------------------------------------

#--------------------------------------
## LOGGING UTILITIES
nfe_counts = Vector{Float64}(undef, EPOCHS + 1)
train_loglikelihood = Vector{Float64}(undef, EPOCHS + 1)
test_loglikelihood = Vector{Float64}(undef, EPOCHS + 1)
train_runtimes = Vector{Float64}(undef, EPOCHS + 1)
inference_runtimes = Vector{Float64}(undef, EPOCHS + 1)
train_runtimes[1] = 0

logger = table_logger(
    [
        "Epoch Number",
        "NFE Count",
        "Train Log Likelihood",
        "Test Log Likelihood",
        "Train Runtime",
        "Inference Runtime",
    ],
    ["Total Loss", "Negative Log Likelihood", "Regularization"],
)
#--------------------------------------

#--------------------------------------
## RECORD DETAILS BEFORE TRAINING STARTS
const dummy_data = train_dataloader.data[:, 1:BATCH_SIZE]
_start_time = time()
_logpx, _r1, _r2, _nfe, _sv = ffjord(dummy_data)
inference_runtimes[1] = time() - _start_time
train_runtimes[1] = 0.0
nfe_counts[1] = _nfe
train_loglikelihood[1] = 0 # data(loglikelihood(ffjord, train_dataloader))
test_loglikelihood[1] = 0 # data(loglikelihood(ffjord, test_dataloader))

logger(
    false,
    Dict(),
    0.0,
    nfe_counts[1],
    train_loglikelihood[1],
    test_loglikelihood[1],
    train_runtimes[1],
    inference_runtimes[1],
)
#--------------------------------------

#--------------------------------------
## WARMSTART THE GRADIENT COMPUTATION
Tracker.gradient(
    p -> loss_function(rand(Float32, 2, 1) |> gpu |> track, ffjord, p; notrack = true),
    ffjord.p,
)
#--------------------------------------

#--------------------------------------
## TRAINING
for epoch = 1:EPOCHS
    λᵣ = λ_func(epoch - 1)
    start_time = time()

    for (i, x) in enumerate(train_dataloader)
        gs = Tracker.gradient(p -> loss_function(x, ffjord, p; λᵣ = λᵣ), ps...)
        update_parameters!(ps, gs, opt)
    end

    # Record the time per epoch
    train_runtimes[epoch+1] = time() - start_time

    # Record the NFE count
    start_time = time()
    _, _, _, nfe, _ = ffjord(dummy_data)
    inference_runtimes[epoch+1] = time() - start_time
    nfe_counts[epoch+1] = nfe

    train_loglikelihood[epoch+1] = 0 # data(loglikelihood(ffjord, train_dataloader))
    test_loglikelihood[epoch+1] = 0 # data(loglikelihood(ffjord, test_dataloader))

    logger(
        false,
        Dict(),
        epoch,
        nfe_counts[epoch+1],
        train_loglikelihood[epoch+1],
        test_loglikelihood[epoch+1],
        train_runtimes[epoch+1],
        inference_runtimes[epoch+1],
    )
end
logger(true, Dict())

# Log the time to generate samples
timings = []
for i = 1:10
    t = time()
    sample(ffjord, ps[1]; nsamples = BATCH_SIZE)
    push!(timings, time() - t)
end
println("Time for Sampling $(BATCH_SIZE) data points: $(minimum(timings)) s")
#--------------------------------------

#--------------------------------------
## STORE THE RESULTS
results = Dict(
    :nfe_counts => nfe_counts,
    :train_likelihood => train_loglikelihood,
    :test_likelihood => test_loglikelihood,
    :train_runtimes => train_runtimes,
    :inference_runtimes => inference_runtimes,
    :sampling_time => minimum(timings),
)

weights = Flux.params(ffjord) .|> cpu .|> untrack
BSON.@save MODEL_WEIGHTS weights

YAML.write_file(FILENAME, results)
#--------------------------------------
