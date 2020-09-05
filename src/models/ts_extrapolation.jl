# Time Series Extrapolation Model
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