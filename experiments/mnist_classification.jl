using RegNeuralODE, OrdinaryDiffEq, Flux, DiffEqFlux, Tracker, TrackerFlux, Random, Plots

Random.seed!(1029)

const epochs = 50
const batch_size = 1204
const lr = 0.01 

train_dataloader, test_dataloader = load_mnist(batch_size)

vanilla_node = ClassifierNODE(
    Chain(flatten, Linear(784, 20, relu)) |> TrackerFlux.track,
    NFECounterNeuralODE(Chain(Linear(20, 10, relu),
                              Linear(10, 10, relu),
                              Linear(10, 20, relu)) |> TrackerFlux.track,
                        [0.f0, 1.f0], Tsit5(),
                        save_everystep = false,
                        reltol = 6f-5, abstol = 6f-5,
                        save_start = false),
    Chain(RegNeuralODE.diffeqsol_to_trackedarray,
          Linear(20, 10)) |> TrackerFlux.track
)

opt_vanilla_node = ADAMW(lr)

reg_node = ClassifierNODE(
    Chain(flatten, Linear(784, 20, relu)) |> TrackerFlux.track,
    NFECounterCallbackNeuralODE(Chain(Linear(20, 10, relu),
                                      Linear(10, 10, relu),
                                      Linear(10, 20, relu)) |> TrackerFlux.track,
                        [0.f0, 1.f0], Tsit5(),
                        save_everystep = false,
                        reltol = 6f-5, abstol = 6f-5,
                        save_start = false),
    Chain(RegNeuralODE.diffeqsol_to_trackedarray,
          Linear(20, 10)) |> TrackerFlux.track
)

# Make the parameters of the network identical
reg_node.p1.data .= vanilla_node.p1.data
reg_node.p2.data .= vanilla_node.p2.data
reg_node.p3.data .= vanilla_node.p3.data

opt_reg_node = ADAMW(lr)

vanilla_node, vanilla_node_logs = RegNeuralODE.train_tracker!(vanilla_node, opt_vanilla_node, epochs,
                                                              train_dataloader, test_dataloader,
                                                              RegNeuralODE.get_loss_function(vanilla_node));

reg_node, reg_node_logs = RegNeuralODE.train_tracker!(reg_node, opt_reg_node, epochs, train_dataloader, test_dataloader,
                                                      RegNeuralODE.get_loss_function(reg_node));