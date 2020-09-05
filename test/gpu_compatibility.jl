using Test
# using Revise, BenchmarkTools

using RegNeuralODE, OrdinaryDiffEq, Flux, DiffEqFlux, Tracker, CUDA

img = rand(Float32, 28, 28, 1, 1) |> gpu;

vanilla_node = ClassifierNODE(
    Chain(flatten, Linear(784, 20, relu)) |> track |> gpu,
    NFECounterNeuralODE(Chain(Linear(20, 10, relu),
                              Linear(10, 10, relu),
                              Linear(10, 20, relu)) |> track |> gpu,
                        [0.f0, 1.f0], Tsit5(),
                        save_everystep = false,
                        reltol = 6f-5, abstol = 6f-5,
                        save_start = false),
    Chain(RegNeuralODE.diffeqsol_to_trackedarray, Linear(20, 10)) |> track |> gpu
)

ps = Flux.trainable(vanilla_node)

gs = Tracker.gradient((p1, p2, p3) -> sum(vanilla_node(img, p1, p2, p3)), ps...)

for (p, g) in zip(ps, gs)
    @test gs.data isa CuArray
    @test size(p) == size(g)
    @test ! all(iszero.(g))
end


reg_node = ClassifierNODE(
    Chain(flatten, Linear(784, 20, relu)) |> track |> gpu,
    NFECounterCallbackNeuralODE(Chain(Linear(20, 10, relu),
                                      Linear(10, 10, relu),
                                      Linear(10, 20, relu)) |> track |> gpu,
                                [0.f0, 1.f0], Tsit5(),
                                save_everystep = false,
                                reltol = 6f-5, abstol = 6f-5,
                                save_start = false),
    Chain(RegNeuralODE.diffeqsol_to_trackedarray, Linear(20, 10)) |> track |> gpu
)

ps = Flux.trainable(reg_node)

# GPU Compilation Fails
gs = Tracker.gradient((p1, p2, p3) -> begin
        sol, sv = reg_node(img, p1, p2, p3)
        return sum(sol) + sum(sv.saveval)
    end, ps...)

for (p, g) in zip(ps, gs)
    @test gs.data isa CuArray
    @test size(p) == size(g)
    @test ! all(iszero.(g))
end