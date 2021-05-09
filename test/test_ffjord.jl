using RegNeuralDE, OrdinaryDiffEq, Flux, CUDA, Tracker
CUDA.allowscalar(false)

mlp_dynamics = TDChain(Dense(3, 10, CUDA.tanh), Dense(11, 2)) |> track |> gpu

x_ = rand(2, 1) |> gpu

# Unregularized FFJORD
ffjord_unreg = TrackedFFJORD(
    mlp_dynamics,
    [0.0f0, 1.0f0],
    false,
    Tsit5(),
    save_everystep = false,
    reltol = 1.4f-3,
    abstol = 1.4f-3,
    save_start = false,
)

## Forward Pass
@code_warntype ffjord_unreg(x_ |> track)
@code_warntype ffjord_unreg(x_ |> track; regularize = true)

## Backward Pass
@code_warntype Tracker.gradient(p -> sum(ffjord_unreg(x_ |> track, p)[1]), ffjord_unreg.p)
@code_warntype Tracker.gradient(
    p -> begin
        logpx, r1, r2, _, _ = ffjord_unreg(x_ |> track, p; regularize = true)
        sum(logpx) + sum(r1) + sum(r2)
    end,
    ffjord_unreg.p,
)

# Regularized FFJORD -- Error Estimates
ffjord_errreg = TrackedFFJORD(
    mlp_dynamics,
    [0.0f0, 1.0f0],
    true,
    Tsit5(),
    save_everystep = false,
    reltol = 1.4f-3,
    abstol = 1.4f-3,
    save_start = false,
)

## Forward Pass
@code_warntype ffjord_errreg(x_ |> track)

## Backward Pass
@code_warntype Tracker.gradient(
    p -> begin
        logpx, _, _, _, sv = ffjord_errreg(x_ |> track, p)
        sum(logpx) + sum(sv.saveval)
    end,
    ffjord_errreg.p,
)
