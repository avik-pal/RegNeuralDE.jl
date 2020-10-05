using RegNeuralODE, OrdinaryDiffEq, Flux, DiffEqFlux, Tracker, Random, Plots

Random.seed!(1029)

epochs = 50
batch_size = 1024
lr = 0.05

train_dataloader, test_dataloader = load_mnist(batch_size, x -> cpu(track(x)))

vanilla_node = ClassifierNODE(
    Chain(flatten, Linear(784, 20)) |> track,
    NFECounterNeuralODE(TDChain(Linear(21, 10, σ),
                                Linear(11, 20)) |> track,
                        [0.f0, 1.f0], Tsit5(),
                        save_everystep = false,
                        reltol = 6f-5, abstol = 6f-5,
                        save_start = false),
    Chain(RegNeuralODE.diffeqsol_to_trackedarray,
          Linear(20, 10)) |> track
)

opt_vanilla_node = ADAMW(lr, (0.9, 0.99), 1e-5)

reg_node = ClassifierNODE(
    Chain(flatten, Linear(784, 20)) |> track,
    NFECounterCallbackNeuralODE(TDChain(Linear(21, 10, σ),
                                        Linear(11, 20)) |> track,
                                [0.f0, 1.f0], Tsit5(),
                                save_everystep = false,
                                reltol = 6f-5, abstol = 6f-5,
                                save_start = false),
    Chain(RegNeuralODE.diffeqsol_to_trackedarray,
          Linear(20, 10)) |> track
)

# Make the parameters of the network identical
reg_node.p1.data .= vanilla_node.p1.data
reg_node.p2.data .= vanilla_node.p2.data
reg_node.p3.data .= vanilla_node.p3.data

opt_reg_node = ADAMW(lr, (0.9, 0.99), 1e-5)

vanilla_node, vanilla_node_logs = RegNeuralODE.train_tracker!(vanilla_node, opt_vanilla_node, epochs,
                                                              train_dataloader, test_dataloader,
                                                              RegNeuralODE.get_loss_function(vanilla_node));

reg_node, reg_node_logs = RegNeuralODE.train_tracker!(reg_node, opt_reg_node, epochs, train_dataloader, test_dataloader,
                                                      RegNeuralODE.get_loss_function(reg_node; λ = 1.0f0));