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