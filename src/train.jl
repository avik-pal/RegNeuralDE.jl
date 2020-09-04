function train!(model::ClassifierNODE, opt, epochs::Int, train_dataloader, test_dataloader, grad_func)
    # Initialize some logging utilities
    nfe_count = Vector{Float64}(undef, epochs)
    train_accuracies = Vector{Float64}(undef, epochs + 1)
    test_accuracies = Vector{Float64}(undef, epochs + 1)
    runtimes = Vector{Float64}(undef, epochs)

    train_accuracies[1] = accuracy(model, train_dataloader)
    test_accuracies[1] = accuracy(model, test_dataloader)
    @printf("Before Training || Train Accuracy: %2.3f || Test Accuracy: %2.3f\n",
            train_accuracies[1], test_accuracies[1])

    ps = Flux.trainable(model)

    for epoch in 1:epochs
        prev_nfe = model.node.nfe[]
        
        start_time = time()

        for (i, (x, y)) in enumerate(train_dataloader)
            gs = grad_func((x, y), ps)
            for (p, g) in zip(ps, gs)
                length(p) == 0 && continue
                Flux.Optimise.update!(opt, data(p), data(g))
            end
        end
        # Record the time per epoch
        runtimes[epoch] = time() - start_time

        # Store the NFE counts
        nfe_count[epoch] = (model.node.nfe[] - prev_nfe) / length(train_dataloader)

        # Compute the train and test accuracies
        train_accuracies[epoch + 1] = accuracy(model, train_dataloader)
        test_accuracies[epoch + 1] = accuracy(model, test_dataloader)
        @printf("Epoch: %3d || Train Accuracy: %2.3f || Test Accuracy: %2.3f || NFE: %2.3f || Epoch Time: %2.3f\n",
                epoch, train_accuracies[epoch + 1], test_accuracies[epoch + 1], nfe_count[epoch], runtimes[epoch])
    end
    
    return (model, Dict(:nfe_count => nfe_count,
                        :train_accuracy => train_accuracies,
                        :test_accuracy => test_accuracies,
                        :time_per_epoch => runtimes))
end


function train_reversediff!(model::ClassifierNODE, opt, epochs::Int,
                            train_dataloader, test_dataloader, loss_fn)
    return train!(model, opt, epochs, train_dataloader, test_dataloader,
                  (data, ps) -> begin
                      return ReverseDiff.gradient(
                          (p1, p2, p3) -> loss_fn(data[1], data[2], model, p1, p2, p3), ps
                      )
                  end)
end

function train_tracker!(model::ClassifierNODE, opt, epochs::Int,
                        train_dataloader, test_dataloader, loss_fn)
    return train!(model, opt, epochs, train_dataloader, test_dataloader,
                  (data, ps) -> begin
                       return Tracker.gradient(
                           (p1, p2, p3) -> loss_fn(data[1], data[2], model, p1, p2, p3), ps...
                       )
                  end)
end


# function train_reversediff!(model::ExtrapolationLatentODE, opt, epochs::Int,
#                             train_dataloader, test_dataloader, loss_fn,
#                             test_loss_func = nothing)
#     test_loss_func = isnothing(test_loss_func) ? loss_fn : test_loss_func 
#     function get_total_loss(data)
#         acc_loss, total_dpoints = 0, 0
#         for (x, t) in data
#             npoints = size(x, 3)
#             total_dpoints += npoints
#             acc_loss = test_loss_func(x, model) * npoints
#         end
#         return acc_loss / total_dpoints
#     end
    
#     running_train_loss = AverageMeter(eltype(train_dataloader.data[1]))
#     # running_test_loss = AverageMeter(eltype(test_dataloader.data[1]))
    
#     nfe_count, train_losses, test_losses = [], [], []

#     push!(train_losses, update!(running_train_loss,
#                                 get_total_loss(train_dataloader)))
#     # push!(test_losses, get_total_loss(test_dataloader))
#     # @printf("Before Training || Train Loss: %2.3f || Test Loss: %2.3f\n",
#     #         train_losses[end], test_losses[end])
#     @printf("Before Training || Train Loss: %2.3f\n", train_losses[end])

#     ps = Flux.trainable(model)
    
#     for epoch in 1:epochs
#         prev_nfe = model.node.nfe[]
    
#         for (x, t) in train_dataloader
#             gs = ReverseDiff.gradient(
#                 (p1, p2, p3) -> loss_fn(x, model, p1, p2, p3), ps
#             )
#             for (p, g) in zip(ps, gs)
#                 length(p) == 0 && continue
#                 Flux.Optimise.update!(opt, p, g)
#             end
#         end

#         # Store the NFE counts
#         push!(nfe_count, (model.node.nfe[] - prev_nfe) / length(train_dataloader))

#         # Compute the test loss
#         push!(train_losses, update!(running_train_loss,
#                                     get_total_loss(train_dataloader)))
#         # push!(test_losses, get_total_loss(test_dataloader))
#         # @printf("Epoch: %3d || Train Loss: %2.3f || Test Loss: %2.3f || NFE: %2.3f\n",
#         #     epoch, train_losses[end], test_losses[end], nfe_count[end])
#         @printf("Epoch: %3d || Train Loss: %2.3f || NFE: %2.3f\n", epoch,
#                 train_losses[end], nfe_count[end])
#     end

#     return model, nfe_count, train_losses, test_losses
# end