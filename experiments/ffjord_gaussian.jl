#--------------------------------------
## LOAD PACKAGES
using RegNeuralODE, OrdinaryDiffEq, Flux, DiffEqFlux, Tracker, Random, Statistics
using YAML, Dates, BSON, NNlib
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
# CUSTOM NN DYNAMICS
function NNlib.σ(x)
    t = CUDA.exp(-abs(x))
    ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
end

NNlib.softplus(x) = ifelse(x > 0, x + CUDA.log1p(CUDA.exp(-x)), CUDA.log1p(CUDA.exp(x)))

_transpose(x) = permutedims(x, (2, 1))

struct ConcatSquashLinear{LW,LB,BW,BB,GW}
    layer_W::LW
    layer_B::LB
    bias_W::BW
    bias_B::BB
    gate_W::GW
end

ConcatSquashLinear(in_dims::Int, out_dims::Int) =
    ConcatSquashLinear(
        Flux.glorot_uniform(out_dims, in_dims),
        zeros(Float32, out_dims, 1),
        Flux.glorot_uniform(out_dims, 1),
        zeros(Float32, out_dims, 1),
        Flux.glorot_uniform(out_dims, 1)
    )

@functor ConcatSquashLinear

(csl::ConcatSquashLinear)(x, t) =
    (csl.layer_W * x .+ csl.layer_B) .* σ.(csl.gate_W * t) .+ (csl.bias_W * t .+ csl.bias_B)

function forw_n_back(csl::ConcatSquashLinear, x, t)
    z1 = csl.layer_W * x .+ csl.layer_B
    z2 = σ.(csl.gate_W * t)
    z3 = csl.bias_W * t .+ csl.bias_B
    r = @. z1 * z2 + z3

    return r, e -> _transpose(csl.layer_W .* z2) * e
end

struct MLPDynamics{LW,LB,BW,BB,GW}
    csl1::ConcatSquashLinear{LW,LB,BW,BB,GW}
    csl2::ConcatSquashLinear{LW,LB,BW,BB,GW}
    csl3::ConcatSquashLinear{LW,LB,BW,BB,GW}
end

@functor MLPDynamics

MLPDynamics(in_dims::Int, hsize::Int) =
    MLPDynamics(
        ConcatSquashLinear(in_dims, hsize),
        ConcatSquashLinear(hsize, hsize),
        ConcatSquashLinear(hsize, in_dims)
    )

function (nn::MLPDynamics)(x, t)
    _t = CUDA.ones(1, 1) .* t
    return nn.csl3(softplus.(nn.csl2(softplus.(nn.csl1(x, _t)), _t)), _t)
end

function forw_n_back(nn::MLPDynamics, x, t, e)
    _t = CUDA.ones(1, 1) .* t
    z1, back1 = forw_n_back(nn.csl1, x, _t)
    tz1 = softplus.(z1)
    z2, back2 = forw_n_back(nn.csl2, tz1, _t)
    tz2 = softplus.(z2)
    z3, back3 = forw_n_back(nn.csl3, tz2, _t)

    return z3, back1(σ.(z1) .* back2(σ.(z2) .* back3(e)))
end
#--------------------------------------

#--------------------------------------
## SETUP THE MODELS + DATASET + TRAINING UTILS
# Get the dataset
train_dataloader, test_dataloader =
    load_gaussian_mixture(BATCH_SIZE, x -> gpu(x), ngaussians = 6, nsamples = 2048)

nn_dynamics = MLPDynamics(2, 16) |> track |> gpu

ffjord = TrackedFFJORD(
    nn_dynamics,
    [0.0f0, 1.0f0],
    true,
    REGULARIZE,
    Tsit5(),
    save_everystep = false,
    reltol = 1.4f-8,
    abstol = 1.4f-8,
    save_start = false,
    dynamics = forw_n_back,
)

ps = Flux.trainable(ffjord)

opt = Flux.Optimise.Optimiser(WeightDecay(1e-5), ADAM(4e-2))

# Anneal the regularization so that it doesn't overpower the
# the main objective
λ₀ = 2.0f3
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
Tracker.gradient(p -> loss_function(dummy_data, ffjord, p; notrack = true), ffjord.p)
#--------------------------------------

#--------------------------------------
## TRAINING
for epoch = 1:EPOCHS
    λᵣ = λ_func(epoch - 1)
    timing = 0

    for (i, x_) in enumerate(train_dataloader)
        x = x_ |> gpu

        start_time = time()
        gs = Tracker.gradient(p -> loss_function(x, ffjord, p; λᵣ = λᵣ), ps...)
        update_parameters!(ps, gs, opt)
        timing += time() - start_time

        x = nothing
        GC.gc(true)
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
    sample(ffjord, 2, ps[1]; nsamples = BATCH_SIZE)
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
