using Plots, Statistics, Random
using Flux, DiffEqFlux, StochasticDiffEq, CUDA, RegNeuralODE, Tracker
using BSON
using Tracker: data
using ColorBrewer
colors = ColorBrewer.palette("Set2", 8)

dataset = BSON.load("data/sde_demo.bson")
sde_data = dataset[:sde_data]
sde_data_vars = dataset[:sde_data_vars]

u0 = reshape(Float32[2.0; 0.0], :, 1)
datasize = 30
tspan = [0.0f0, 1.0f0]
tsteps = range(tspan[1], tspan[2], length = datasize) |> track

plots1, plots2 = [], []
ptime1, ptime2 = [], []
ttime1, ttime2 = [], []
loss1, loss2 = [], []
nfe1, nfe2 = [], []

function loss_function(u0, p, i)
    sol, nfe1, nfe2, sv = neuralsde(u0, p)
    means, vars = mean(sol; dims = 3), var(sol; dims = 3)
    l2_means = mean(abs2, sde_data .- means)
    l2_vars = mean(abs2, sde_data_vars .- vars)
    reg = REGULARIZE ? 0.2f0 * sum(sv.saveval) : 0.0f0
    loss = l2_means + l2_vars
    if i % 50 == 0
        @show i, data(loss), data(l2_means), data(l2_vars), data(reg), nfe1, nfe2
    end
    i == -1 && return (loss, nfe1)
    return loss + reg
end

for i in 1:3
    for reg in [true, false]
        Random.seed!(i)

        drift_dudt = Chain(x -> x .^ 3, Dense(2, 50, tanh), Dense(50, 2)) |> track
        diffusion_dudt = Dense(2, 2) |> track

        REGULARIZE = reg

        neuralsde = TrackedNeuralDSDE(
            drift_dudt,
            diffusion_dudt,
            [0.0f0, 1.2f0],
            REGULARIZE,
            SOSRI(),
            saveat = tsteps,
            reltol = 3f-1,
            abstol = 3f-1,
        )

        u0_ = repeat(u0, 1, 10)
        loss_function(u0_ |> track, neuralsde.p, 0)
        Tracker.gradient(p -> loss_function(u0_ |> track, p, 0), neuralsde.p)

        opt = AdaBelief(0.025)
        ps = neuralsde.p
        _t = time()
        for iter in 1:100
            gs = Tracker.gradient(p -> loss_function(u0_ |> track, p, iter), ps)[1]
            update_parameters!((ps,), (gs,), opt)
        end
        total_time = time() - _t

        u0_ = repeat(u0, 1, 100)
        ps = neuralsde.p
        _t = time()
        for iter in 1:150
            gs = Tracker.gradient(p -> loss_function(u0_ |> track, p, iter), ps)[1]
            update_parameters!((ps,), (gs,), opt)
        end
        total_time += time() - _t
        u0_ = u0_ |> track
        ptime = @belapsed neuralsde(u0_, ps)[1]

        begin
            u0_ = repeat(u0, 1, 256) |> track
            ptime = @belapsed neuralsde(u0_, ps)[1]
            preds = @btime neuralsde(u0_, ps)[1]
            preds = preds |> untrack
            means, vars = mean(preds; dims = 3), var(preds, dims = 3)

            plt = plot(
                tsteps |> untrack,
                means[1, :, 1],
                ribbon = vars[1, :, 1],
                color = colors[1],
                fillalpha = 0.2,
                label = "Pred: Dim 1",
                linewidth = 3,
                legend = :bottomleft
            )
            plot!(
                plt,
                tsteps |> untrack,
                means[2, :, 1],
                ribbon = vars[2, :, 1],
                color = colors[2],
                fillalpha = 0.2,
                label = "Pred: Dim 2",
                linewidth = 3
            )
            scatter!(
                plt,
                tsteps |> untrack,
                sde_data[1, :],
                yerror = sde_data_vars[1, :],
                label = "Data: Dim 1",
                color = colors[1],
                msc = colors[1],
            )
            scatter!(
                plt,
                tsteps |> untrack,
                sde_data[2, :],
                yerror = sde_data_vars[2, :],
                label = "Data: Dim 2",
                color = colors[2],
                msc = colors[2]
            )

            if REGULARIZE
                push!(plots1, plt)
            else
                push!(plots2, plt)
            end
        end

        if REGULARIZE
            push!(ttime1, total_time)
            l, n = loss_function(u0_ |> track, neuralsde.p, -1)
            push!(loss1, l)
            push!(ptime1, ptime)
            push!(nfe1, n)
        else
            push!(ttime2, total_time)
            l, n = loss_function(u0_ |> track, neuralsde.p, -1)
            push!(loss2, l)
            push!(ptime2, ptime)
            push!(nfe2, n)
        end
    end
end

mean(loss1), std(loss1), mean(loss2), std(loss2)
mean(nfe1), std(nfe1), mean(nfe2), std(nfe2)
mean(ptime1), std(ptime1), mean(ptime2), std(ptime2)
mean(ttime1), std(ttime1), mean(ttime2), std(ttime2)


title!(plots2[3], "Unregularized Neural SDE")
title!(plots1[1], "Regularized Neural SDE")
xlabel!(plots1[1], "t")
plt = plot(plots2[3], plots1[1], layout = (2, 1))

savefig(plt, "spiral_sde.pdf")