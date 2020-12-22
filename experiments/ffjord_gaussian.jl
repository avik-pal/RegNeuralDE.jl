#--------------------------------------
## LOAD PACKAGES
using RegNeuralODE, OrdinaryDiffEq, Flux, DiffEqFlux, Tracker, Random, Statistics
using ProgressLogging, YAML, Dates, BSON
using CUDA
using RegNeuralODE: loglikelihood
using Flux.Optimise: update!
using Flux: @functor, glorot_uniform, logitcrossentropy
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

DATA_PATH = config["data_path"]
hparams = config["hyperparameters"]
BATCH_SIZE = hparams["batch_size"]
REGULARIZE = hparams["regularize"]
HIDDEN_DIMS = hparams["hidden_dims"]
INPUT_DIMS = hparams["input_dims"]
LR = hparams["lr"]
EPOCHS = hparams["epochs"]
EXPERIMENT_LOGDIR = joinpath(config["log_dir"], "$(string(now()))_$REGULARIZE")
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
    load_multimodel_gaussian(BATCH_SIZE, ngaussians = 2, nsamples = 200)

nn_dynamics = MLPDynamics(INPUT_DIMS, HIDDEN_DIMS, HIDDEN_DIMS) |> gpu
ffjord =
    TrackedFFJORD(
        nn_dynamics |> track,
        [0.0f0, 1.0f0],
        REGULARIZE,
        INPUT_DIMS,
        Vern7(),
        save_everystep = false,
        reltol = 1.4f-8,
        abstol = 1.4f-8,
        save_start = false,
    ) |> track

if REGULARIZE
    function loss_function(x, model, p; λ = 1.0f2)
        pred, sv, sol = model(x, p)
        return -mean(pred) + λ * sum(sv.saveval)
    end
else
    function loss_function(x, model, p)
        pred, _, _, sol = model(x, p)
        return -mean(pred)
    end
end
#--------------------------------------

nfe_counts = Vector{Float64}(undef, EPOCHS + 1)
train_loglikelihood = Vector{Float64}(undef, EPOCHS + 1)
test_loglikelihood = Vector{Float64}(undef, EPOCHS + 1)
train_runtimes = Vector{Float64}(undef, EPOCHS + 1)  # The first value is a dummy value
inference_runtimes = Vector{Float64}(undef, EPOCHS + 1)

train_runtimes[1] = 0

ps = ffjord.p

dummy_data = rand(Float32, 2, 32) |> track
start_time = time()
# Warmup
ffjord(dummy_data)
sol = ffjord(dummy_data)[end]
inference_runtimes[1] = time() - start_time
train_runtimes[1] = 0.0
nfe_counts[1] = sol.destats.nf
train_loglikelihood[1] = data(loglikelihood(ffjord, train_dataloader))
test_loglikelihood[1] = data(loglikelihood(ffjord, test_dataloader))
@info (
    train_runtimes[1],
    inference_runtimes[1],
    nfe_counts[1],
    train_loglikelihood[1],
    test_loglikelihood[1],
)

opt = ADAM(LR)

@progress for epoch = 1:EPOCHS
    start_time = time()

    @progress for (i, x) in enumerate(train_dataloader)
        gs = Tracker.gradient(p -> loss_function(x, ffjord, p), ps)[1]
        update!(opt, data(ps), data(gs))
    end
    # Record the time per epoch
    train_runtimes[epoch+1] = time() - start_time

    # Record the NFE count
    start_time = time()
    sol = ffjord(dummy_data)[end]
    inference_runtimes[epoch+1] = time() - start_time
    nfe_counts[epoch+1] = sol.destats.nf

    train_loglikelihood[epoch+1] = data(loglikelihood(ffjord, train_dataloader))
    test_loglikelihood[epoch+1] = data(loglikelihood(ffjord, test_dataloader))
    @info (
        train_runtimes[epoch+1],
        inference_runtimes[epoch+1],
        nfe_counts[epoch+1],
        train_loglikelihood[epoch+1],
        test_loglikelihood[epoch+1],
    )
end

results = Dict(
    :nfe_counts => nfe_counts,
    :train_likelihood => train_likelihood,
    :test_likelihood => test_likelihood,
    :train_runtimes => train_runtimes,
    :inference_runtimes => inference_runtimes,
)

BSON.@save MODEL_WEIGHTS Dict(:p => ffjord.p)

YAML.write_file(FILENAME, results)
