# MLP Dynamics conditioned on the Time Step
struct TimeDependentMLPDynamics{M}
    model::M
    
    function TimeDependentMLPDynamics(dims::Vector{Int}, act = tanh)
        layers = []
        for i in 1:length(dims) - 2
            push!(layers, Dense(dims[i] + 1, dims[i + 1], act))
        end
        push!(layers, Dense(dims[end - 1] + 1, dims[end]))
        model = Chain(layers...)
        return new{typeof(model)}(model)
    end
end

Flux.@functor TimeDependentMLPDynamics

function (m::TimeDependentMLPDynamics)(x, t::Number)
    _t = similar(x, 1, size(x, 2))
    fill!(_t, t)
    for layer in m.model
        x = layer(cat(x, _t, dims = 1))
    end
    return x
end

# Neural ODE Variants
struct NFECounterNeuralODE{M, P, RE, T, A, K} <: DiffEqFlux.NeuralDELayer
    model::M
    p::P
    re::RE
    tspan::T
    args::A
    kwargs::K
    nfe::Vector{Int}

    function NFECounterNeuralODE(model, tspan, args...; p = nothing,
                                 kwargs...)
        _p, re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        new{typeof(model), typeof(p), typeof(re),
            typeof(tspan), typeof(args), typeof(kwargs)}(
            model, p, re, tspan, args, kwargs, [0])
    end
end

function (n::NFECounterNeuralODE)(x, p = n.p)
    function dudt_(u, p, t)
        n.nfe[] += 1
        n.re(p)(u)
    end
    ff = ODEFunction{false}(dudt_, tgrad = DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, n.tspan, p)
    solve(prob, n.args...; sensealg = SensitivityADPassThrough(), n.kwargs...)
end

function (n::NFECounterNeuralODE{M})(x, p = n.p) where M <: TimeDependentMLPDynamics
    function dudt_(u, p, t)
        n.nfe[] += 1
        n.re(p)(u, t)
    end
    ff = ODEFunction{false}(dudt_, tgrad = DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, n.tspan, p)
    solve(prob, n.args...; sensealg = SensitivityADPassThrough(), n.kwargs...)
end


struct NFECounterCallbackNeuralODE{M, P, RE, T, A, K} <: DiffEqFlux.NeuralDELayer
    model::M
    p::P
    re::RE
    tspan::T
    args::A
    kwargs::K
    nfe::Vector{Int}

    function NFECounterCallbackNeuralODE(model, tspan, args...; p = nothing,
                                         kwargs...)
        _p, re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        new{typeof(model), typeof(p), typeof(re),
            typeof(tspan), typeof(args), typeof(kwargs)}(
            model, p, re, tspan, args, kwargs, [0])
    end
end

function (n::NFECounterCallbackNeuralODE{M, P})(x, p::P = n.p) where {M, P<:TrackedArray}
    function dudt_(u, p, t)
        n.nfe[] += 1
        n.re(p)(u)
    end

    tspan = Tracker.collect(eltype(p).(n.tspan))

    sv = SavedValues(eltype(tspan), eltype(p))
    svcb = SavingCallback(
        (u, t, integrator) -> integrator.EEst * integrator.dt, sv
    )

    ff = ODEFunction{false}(dudt_)
    prob = ODEProblem{false}(ff, x, tspan, p)

    solve(prob, n.args...; sensealg = SensitivityADPassThrough(),
          callback = svcb, n.kwargs...), sv
end

function (n::NFECounterCallbackNeuralODE)(x, p = n.p)
    function dudt_(u, p, t)
        n.nfe[] += 1
        n.re(p)(u)
    end

    tspan = eltype(p).(n.tspan)

    sv = SavedValues(eltype(tspan), eltype(p))
    svcb = SavingCallback(
        (u, t, integrator) -> integrator.EEst * integrator.dt, sv
    )

    ff = ODEFunction{false}(dudt_, tgrad=DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, tspan, p)

    solve(prob, n.args...; sensealg = SensitivityADPassThrough(),
          callback = svcb, n.kwargs...), sv
end

function (n::NFECounterCallbackNeuralODE{M})(x, p::AbstractArray{T} = n.p) where {T, M <: TimeDependentMLPDynamics}
    function dudt_(u, p, t)
        n.nfe[] += 1
        n.re(p)(u, t)
    end

    tspan = T.(n.tspan)

    sv = SavedValues(eltype(tspan), T)
    svcb = SavingCallback(
        (u, t, integrator) -> integrator.EEst * integrator.dt, sv
    )

    ff = ODEFunction{false}(dudt_, tgrad=DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, tspan, p)

    solve(prob, n.args...; sensealg = SensitivityADPassThrough(),
          callback = svcb, n.kwargs...), sv
end


# Classification Network mostly useful for ReverseDiff
struct ClassifierNODE{N, P1, P2, T}
    preode::P1
    node::N
    postode::P2
    p1::T
    p2::T
    p3::T
    
    function ClassifierNODE(preode, node, postode)
        p1, re1 = Flux.destructure(preode)
        p2 = node.p
        p3, re3 = Flux.destructure(postode)
        return new{typeof(node), typeof(re1), typeof(re3),
                   typeof(p2)}(re1, node, re3, p1, p2, p3)
    end
end

Flux.trainable(m::ClassifierNODE) = (m.p1, m.p2, m.p3)

function (m::ClassifierNODE)(x, p1 = m.p1, p2 = m.p2, p3 = m.p3)
    x = m.preode(p1)(x)
    x = m.node(x, p2)
    return m.postode(p3)(x)
end

function (m::ClassifierNODE{T})(x, p1 = m.p1, p2 = m.p2,
                                p3 = m.p3) where T<:NFECounterCallbackNeuralODE 
    x = m.preode(p1)(x)
    x, sv = m.node(x, p2)
    return m.postode(p3)(x), sv
end


# Time Series Extrapolation Model

## Recognition RNN for mapping from time series data to a latent vector
struct RecognitionRNN{P, Q, R}
    i2h::Dense{P, Q, R}
    h2o::Dense{P, Q, R}
end

Flux.@functor RecognitionRNN

RecognitionRNN(;latent_dim::Int, nhidden::Int, obs_dim::Int) =
    RecognitionRNN(Dense(obs_dim + nhidden, nhidden),
                   Dense(nhidden, latent_dim * 2))

function (rrnn::RecognitionRNN)(x, h)
    h = tanh.(rrnn.i2h(vcat(x, h)))
    return rrnn.h2o(h), h
end

## LatentODE model
struct ExtrapolationLatentODE{N, R, D, S, T}
    rrnn::R
    node::N
    dec::D
    sol_to_arr::S
    p1::T
    p2::T
    p3::T
    latent_dim::Int
    obs_dim::Int
    nhidden_ode::Int
    nhidden_rnn::Int
    
    function ExtrapolationLatentODE(latent_dim::Int, obs_dim::Int, nhidden_ode::Int,
                                    nhidden_rnn::Int, node_func, sol_to_arr)
        rrnn = RecognitionRNN(;latent_dim = latent_dim, nhidden = nhidden_rnn,
                              obs_dim = obs_dim)
        dec = Chain(Dense(latent_dim, nhidden_ode, relu),
                    Dense(nhidden_ode, obs_dim))
        latent_ode_func = Chain(Dense(latent_dim, nhidden_ode, elu),
                                Dense(nhidden_ode, nhidden_ode, elu),
                                Dense(nhidden_ode, latent_dim))
        node = node_func(latent_ode_func)
        
        p1, rrnn = Flux.destructure(rrnn)
        p2 = node.p
        p3, dec = Flux.destructure(dec)
        
        return new{typeof(node), typeof(rrnn), typeof(dec), typeof(sol_to_arr),
                   typeof(p2)}(rrnn, node, dec, sol_to_arr, p1, p2, p3, latent_dim,
                   obs_dim, nhidden_ode, nhidden_rnn)
    end
end

Flux.trainable(m::ExtrapolationLatentODE) = (m.p1, m.p2, m.p3)

function (m::ExtrapolationLatentODE)(x, p1 = m.p1, p2 = m.p2, p3 = m.p3)
    # x --> D x T x B
    rrnn = m.rrnn(p1)
    hs = zeros(eltype(x), m.nhidden_rnn, size(x, 3))
    for t in reverse(2:size(x, 2))
        _, hs = rrnn(x[:, t, :], hs)
    end
    out, _ = rrnn(x[:, 1, :], hs)
    
    latent_vec = out
    
    qz0_μ = latent_vec[1:m.latent_dim, :]
    qz0_logvar = latent_vec[(m.latent_dim + 1):end, :]
    
    # Reparameterization Trick for Sampling from Normal Distribution
    z0 = randn(eltype(x), size(qz0_μ)) .* exp.(qz0_logvar ./ 2) .+ qz0_μ
    
    n_ode_sol = m.sol_to_arr(m.node(z0, p2))
    return reshape(m.dec(p3)(reshape(permutedims(n_ode_sol, (1, 3, 2)),
                                     m.latent_dim, :)),
                   size(x, 1), :, size(z0, 2)), qz0_μ, qz0_logvar
end


function (m::ExtrapolationLatentODE{T})(x, p1 = m.p1, p2 = m.p2,
                                        p3 = m.p3) where T<:NFECounterCallbackNeuralODE
    # x --> D x T x B
    rrnn = m.rrnn(p1)
    hs = zeros(eltype(x), m.nhidden_rnn, size(x, 3))
    for t in reverse(2:size(x, 2))
        _, hs = rrnn(x[:, t, :], hs)
    end
    out, _ = rrnn(x[:, 1, :], hs)
    
    latent_vec = out
    
    qz0_μ = latent_vec[1:m.latent_dim, :]
    qz0_logvar = latent_vec[(m.latent_dim + 1):end, :]
    
    # Reparameterization Trick for Sampling from Normal Distribution
    z0 = randn(eltype(x), size(qz0_μ)) .* exp.(qz0_logvar ./ 2) .+ qz0_μ
    
    n_ode_sol, sv = m.node(z0, p2)
    n_ode_sol = m.sol_to_arr(n_ode_sol)
    return reshape(m.dec(p3)(reshape(permutedims(n_ode_sol, (1, 3, 2)),
                                     m.latent_dim, :)),
                   size(x, 1), :, size(z0, 2)), qz0_μ, qz0_logvar, sv
end


# Latent GRU Model
struct PhysionetLatentGRU{U, R, N, D}
    update_gate::U
    reset_gate::R
    new_state::N
    latent_dim::D

    function PhysionetLatentGRU(in_dim::Int, h_dim::Int,
                                latent_dim::Int)
        update_gate = Chain(
            Dense(latent_dim * 2 + in_dim, h_dim, tanh),
            Dense(h_dim, latent_dim, σ))
        reset_gate = Chain(
            Dense(latent_dim * 2 + in_dim, h_dim, tanh),
            Dense(h_dim, latent_dim, σ))
        new_state = Chain(
            Dense(latent_dim * 2 + in_dim, h_dim, tanh),
            Dense(h_dim, latent_dim * 2, σ))
        return new{typeof(update_gate), typeof(reset_gate),
                   typeof(new_state), typeof(latent_dim)}(
                   update_gate, reset_gate, new_state, latent_dim)
    end
end

Flux.@functor PhysionetLatentGRU

function (p::PhysionetLatentGRU)(y_mean, y_std, x)
    # x is the concatenation of a data and mask
    y_concat = cat(y_mean, y_std, x, dims = 1)
    
    update_gate = p.update_gate(y_concat)
    reset_gate = p.reset_gate(y_concat)
    
    concat = cat(y_mean .* reset_gate, y_std .* reset_gate,
                 x, dims = 1)
    
    new_state = p.new_state(concat)
    new_state_mean = new_state[1:p.latent_dim, :]
    # The easy-neural-ode paper uses abs, which is indeed a strange
    # choice. The standard is to predict log_std and take exp.
    new_state_std = abs.(new_state[p.latent_dim + 1:end, :])
    
    new_y_mean = @. (1 - update_gate) * new_state_mean + update_gate * y_mean
    new_y_std = @. (1 - update_gate) * new_state_std + update_gate * y_std
    
    mask = reshape(sum(x[size(x, 1) ÷ 2:end, :], dims=1) .> 0, 1, :)

    new_y_mean = @. mask * new_y_mean + (1 - mask) * y_mean
    new_y_std = @. abs(mask * new_y_std + (1 - mask) * y_std)
    
    return new_y_mean, new_y_std
end