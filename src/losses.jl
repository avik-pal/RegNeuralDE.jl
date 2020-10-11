# function get_loss_function(model::NFECounterFFJORD, regularize = false)
#     function ffjord_logpx_loss(x::AbstractArray{T}, model, p) where T
#         logpx, λ₁, λ₂ = model(x, p; regularize = regularize)
#         return mean(-logpx .+ T(0.01) .* λ₁ .+ λ₂)
#     end
#     return ffjord_logpx_loss
# end

# function get_loss_function(model::NFECounterCallbackFFJORD; λ = 1.0f2)
#     function ffjord_regularized_logpx_loss(x, model, p)
#         logpx, sv = model(x, p)
#         return -mean(logpx) + λ * mean(sv.saveval)
#     end
#     return ffjord_regularized_logpx_loss
# end

# function get_loss_function(model::ExtrapolationLatentODE)
#     function loss_extrapolating_latent_ode(x::AbstractArray{T},
#                                            model, ps...) where T
#         pred, qz0_μ, qz0_logvar = model(x, ps...)
#         logpx = reshape(sum(log_normal_pdf(x, pred, T(2 * log(0.3))),
#                             dims=(1, 2)), :)
#         analytic_kl = reshape(sum(normal_kl(qz0_μ, qz0_logvar, zeros(T, size(qz0_μ)),
#                                             zeros(T, size(qz0_logvar))),
#                                   dims = 1), :)

#         return mean(analytic_kl .- logpx)
#     end
#     return loss_extrapolating_latent_ode
# end

# function get_loss_function(model::ExtrapolationLatentODE{M}; λ = 1.0f2) where M<:NFECounterCallbackNeuralODE
#     function regularized_loss_extrapolating_latent_ode(x::AbstractArray{T},
#                                                        model, ps...) where T
#         pred, qz0_μ, qz0_logvar, sv = model(x, ps...)
#         logpx = reshape(sum(log_normal_pdf(x, pred, T(2 * log(0.3))),
#                             dims=(1, 2)), :)
#         analytic_kl = reshape(sum(normal_kl(qz0_μ, qz0_logvar, zeros(T, size(qz0_μ)),
#                                             zeros(T, size(qz0_logvar))),
#                                   dims = 1), :)

#         return mean(analytic_kl .- logpx) + λ * mean(sv.saveval)
#     end
#     return regularized_loss_extrapolating_latent_ode
# end