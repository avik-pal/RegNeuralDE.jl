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
config_file = joinpath(pwd(), "experiments", "configs", "ffjord_tabular.yml")
config = YAML.load_file(config_file)

Random.seed!(config["seed"])

hparams = config["hyperparameters"]
const BATCH_SIZE = hparams["batch_size"]
const REGULARIZE = hparams["regularize"]
EPOCHS = hparams["epochs"]
EXPERIMENT_LOGDIR =
    joinpath(pwd(), "results", "ffjord_tabular", "$(string(now()))_$REGULARIZE")
MODEL_WEIGHTS = joinpath(EXPERIMENT_LOGDIR, "weights.bson")
FILENAME = joinpath(EXPERIMENT_LOGDIR, "results.yml")

# Create a directory to store the results
isdir(EXPERIMENT_LOGDIR) || mkpath(EXPERIMENT_LOGDIR)
cp(config_file, joinpath(EXPERIMENT_LOGDIR, "config.yml"))
#--------------------------------------

#--------------------------------------
## NEURAL NETWORK
struct MLPDynamics{T}
    W1::T
    W2::T
    W3::T
    B1::T
    B2::T
    B3::T
end

@functor MLPDynamics

function MLPDynamics(dims::Int, hsize::Int)
    return MLPDynamics(
        Flux.glorot_uniform(hsize, dims),
        Flux.glorot_uniform(hsize, hsize),
        Flux.glorot_uniform(dims, hsize),
        zeros(Float32, hsize, 1),
        zeros(Float32, hsize, 1),
        zeros(Float32, dims, 1),
    )
end

(m::MLPDynamics)(x) = m.W3 * CUDA.tanh.(m.W2 * CUDA.tanh.(m.W1 * x .+ m.B1) .+ m.B2) .+ m.B3

_transpose(x) = permutedims(x, (2, 1))

function forw_n_back(m::MLPDynamics, x, t, e)
    # x -> N x B, e -> N x B
    z1 = m.W1 * x .+ m.B1           # H x B
    tz1 = CUDA.tanh.(z1)            # H x B
    dtz1 = @. 1 - CUDA.pow(tz1, 2)  # H x B
    z2 = m.W2 * tz1 .+ m.B2         # H x B
    tz2 = CUDA.tanh.(z2)            # H x B
    dtz2 = @. 1 - CUDA.pow(tz2, 2)  # H x B
    z3 = m.W3 * tz2 .+ m.B3         # N x B

    eJ = _transpose(m.W1) * (dtz1 .* (_transpose(m.W2) * (dtz2 .* (_transpose(m.W3) * e))))
    return z3, eJ
end
#--------------------------------------

#--------------------------------------
## SETUP THE MODELS + DATASET + TRAINING UTILS
# Get the dataset
train_dataloader, test_dataloader =
    load_miniboone(BATCH_SIZE, "data/miniboone.npy", 0.8, x -> cpu(x))

# Leads to Spurious type promotion needs to be fixed before usage
const nn_dynamics = MLPDynamics(43, 860) |> gpu |> track
# nn_dynamics =
#     TDChain(Dense(44, 100, CUDA.tanh), Dense(101, 100, CUDA.tanh), Dense(101, 43)) |>
#     gpu |>
#     track

const ffjord = TrackedFFJORD(
    nn_dynamics,
    [0.0f0, 1.0f0],
    false,
    REGULARIZE,
    Tsit5(),
    save_everystep = false,
    reltol = 1.4f-8,
    abstol = 1.4f-8,
    save_start = false,
    dynamics = forw_n_back,
)

ps = Flux.trainable(ffjord)

opt = Flux.Optimise.Optimiser(WeightDecay(1e-5), ADAM(4e-3))

# Anneal the regularization so that it doesn't overpower the
# the main objective
λ₀ = 5.0f3
λ₁ = 1.0f3
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
const dummy_data = train_dataloader.data[:, 1:BATCH_SIZE] |> gpu
_start_time = time()
_logpx, _r1, _r2, _nfe, _sv = ffjord(dummy_data)
inference_runtimes[1] = time() - _start_time
train_runtimes[1] = 0.0
nfe_counts[1] = _nfe
train_loglikelihood[1] = data(loglikelihood(ffjord, train_dataloader))
test_loglikelihood[1] = data(loglikelihood(ffjord, test_dataloader))

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
    p -> loss_function(dummy_data, ffjord, p; notrack = true),
    ffjord.p,
)
#--------------------------------------

#--------------------------------------
## TRAINING
for epoch = 1:EPOCHS
    λᵣ = λ_func(epoch - 1)
    timing = 0

    for (i, x_) in enumerate(train_dataloader)
        x = x_ |> gpu

        start_time = time()
        gs = Tracker.gradient(p -> loss_function(x, ffjord, p; λᵣ = λᵣ, notrack = false), ps...)
        update_parameters!(ps, gs, opt)
        timing += time() - start_time

        x = nothing
    end

    # Record the time per epoch
    train_runtimes[epoch+1] = timing

    # Record the NFE count
    start_time = time()
    _, _, _, nfe, _ = ffjord(dummy_data)
    inference_runtimes[epoch+1] = time() - start_time
    nfe_counts[epoch+1] = nfe

    train_loglikelihood[epoch+1] = data(loglikelihood(ffjord, train_dataloader))
    test_loglikelihood[epoch+1] = data(loglikelihood(ffjord, test_dataloader))

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
    sample(ffjord, 43, ps[1]; nsamples = BATCH_SIZE)
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
