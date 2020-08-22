function train!(model::ClassifierNODE, opt, epochs::Int, train_dataloader,
                test_dataloader, loss_fn)
    nfe_count, train_accuracies, test_accuracies = [], [], []

    push!(train_accuracies, accuracy(model, train_dataloader))
    push!(test_accuracies, accuracy(model, test_dataloader))
    @printf("Before Training || Train Accuracy: %2.3f || Test Accuracy: %2.3f\n",
            train_accuracies[end], test_accuracies[end])
    
    ps = Flux.trainable(model)
    
    for epoch in 1:epochs
        prev_nfe = model.node.nfe[]
    
        for (i, (x, y)) in enumerate(train_dataloader)
            gs = ReverseDiff.gradient(
                (p1, p2, p3) -> loss_fn(x, y, model, p1, p2, p3), ps
            )
            for (p, g) in zip(ps, gs)
                length(p) == 0 && continue
                Flux.Optimise.update!(opt, p, g)
            end
            
            if i % 10 == 0
                @printf("Iterations Completed: [%3d / %3d]\n", i, length(train_dataloader))
            end
        end

        # Store the NFE counts
        push!(nfe_count, (model.node.nfe[] - prev_nfe) / length(train_dataloader))

        # Compute the train and test accuracies
        push!(train_accuracies, accuracy(model, train_dataloader))
        push!(test_accuracies, accuracy(model, test_dataloader))
        @printf("Epoch: %3d || Train Accuracy: %2.3f || Test Accuracy: %2.3f || NFE: %2.3f\n",
                epoch, train_accuracies[end], test_accuracies[end], nfe_count[end])
    end

    return model, nfe_count, train_accuracies, test_accuracies
end