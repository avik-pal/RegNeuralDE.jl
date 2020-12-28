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

# Scalar indexing issue is currently unresolved
CUDA.allowscalar(true)
#--------------------------------------

#--------------------------------------
## CONFIGURATION
## Training Parameters
config_file = joinpath(pwd(), "experiments", "configs", "ffjord_gaussian.yml")
config = YAML.load_file(config_file)

Random.seed!(config["seed"])

hparams = config["hyperparameters"]
BATCH_SIZE = hparams["batch_size"]
REGULARIZE = hparams["regularize"]
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
## NEURAL NETWORK
struct ConcatSquashLayer{L,B,G}
    linear::L
    hyper_bias::B
    hyper_gate::G
end

function ConcatSquashLayer(in_dim::Int, out_dim::Int)
    l = Dense(in_dim, out_dim)
    b = Dense(1, out_dim)
    g = Dense(1, out_dim)
    return ConcatSquashLayer(l, b, g)
end

@functor ConcatSquashLayer

function (csl::ConcatSquashLayer)(x, t)
    _t = CUDA.ones(Float32, 1, size(x, 2)) .* t
    return csl.linear(x) .* σ.(csl.hyper_gate(_t)) .+ csl.hyper_bias(_t)
end

cusoftplus(x) = CUDA.log(CUDA.exp(x) + 1)

struct MLPDynamics{L1,L2,L3}
    l1::L1
    l2::L2
    l3::L3
end

MLPDynamics(in_dims::Int, hdim1::Int, hdim2::Int) = MLPDynamics(
    ConcatSquashLayer(in_dims, hdim1),
    ConcatSquashLayer(hdim1, hdim2),
    ConcatSquashLayer(hdim2, in_dims),
)

@functor MLPDynamics

function (mlp::MLPDynamics)(x, t)
    x = cusoftplus.(mlp.l1(x, t))
    x = cusoftplus.(mlp.l2(x, t))
    return mlp.l3(x, t)
end
#--------------------------------------

#--------------------------------------
## SETUP THE MODELS + DATASET + TRAINING UTILS
# Get the dataset
train_dataloader, test_dataloader =
    load_multimodel_gaussian(BATCH_SIZE, x -> gpu(track(x)), ngaussians = 6, nsamples = 4096)

nn_dynamics = MLPDynamics(2, 32, 32) |> gpu |> track
ffjord = TrackedFFJORD(
    nn_dynamics,
    [0.0f0, 1.0f0],
    REGULARIZE,
    2,
    Tsit5(),
    save_everystep = false,
    reltol = 1.4f-3,
    abstol = 1.4f-3,
    save_start = false,
)

ps = Flux.trainable(ffjord)

opt = Flux.Optimise.Optimiser(WeightDecay(1e-5), ADAM(4e-2))

# Anneal the regularization so that it doesn't overpower the
# the main objective
λ₀ = 1.0f3
λ₁ = 1.0f1
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
dummy_data = CUDA.rand(Float32, 2, BATCH_SIZE) |> track
start_time = time()
_logpx, _r1, _r2, _nfe, _sv = ffjord(dummy_data)
inference_runtimes[1] = time() - start_time
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
        for (p, g) in zip(ps, gs)
            length(p) == 0 && continue
            update!(opt, data(p), data(g))
        end
    end

    # Record the time per epoch
    train_runtimes[epoch+1] = time() - start_time

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
for i in 1:100
    t = time()
    sample(ffjord, ps[1]; nsamples = BATCH_SIZE)
    push!(timings, time() - t)
end
print("Time for Sampling $(BATCH_SIZE) data points: $(minimum(timings)) s")
#--------------------------------------

#--------------------------------------
## STORE THE RESULTS
results = Dict(
    :nfe_counts => nfe_counts,
    :train_likelihood => train_likelihood,
    :test_likelihood => test_likelihood,
    :train_runtimes => train_runtimes,
    :inference_runtimes => inference_runtimes,
    :sampling_time => minimum(timings),
)

weights = Flux.params(ffjord) .|> cpu .|> untrack
BSON.@save MODEL_WEIGHTS weights

YAML.write_file(FILENAME, results)
#--------------------------------------
