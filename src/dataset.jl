function load_mnist(batchsize::Int, device_func = cpu)
    # Use MLDataUtils LabelEnc for natural onehot conversion
    onehot(labels_raw) = convertlabel(LabelEnc.OneOfK, labels_raw,
                                      LabelEnc.NativeLabels(collect(0:9)))
    # Load MNIST
    imgs, labels_raw = MNIST.traindata();
    # Process images into (H,W,C,BS) batches
    x_train_data = Float32.(reshape(imgs, size(imgs,1), size(imgs,2), 1, size(imgs,3)))
    y_train_data = onehot(labels_raw)
    # Load MNIST Test
    imgs, labels_raw = MNIST.testdata();
    # Process images into (H,W,C,BS) batches
    x_test_data = Float32.(reshape(imgs, size(imgs,1), size(imgs,2), 1, size(imgs,3)))
    y_test_data = onehot(labels_raw)
    return (
        # Use Flux's DataLoader to automatically minibatch and shuffle the data
        DataLoader(device_func.(collect.([x_train_data, y_train_data]))...;
                   batchsize = batchsize, shuffle = true),
        # Don't shuffle the test data
        DataLoader(device_func.(collect.([x_test_data, y_test_data]))...;
                   batchsize = batchsize, shuffle = false)
    )
end


function load_physionet(batchsize::Int, path::String, train_test_split::Float64 = 0.8,
                        device_func = cpu)
    BSON.@load path data
    total_obs = size(data[:observed_data])[end]
    train_idx, test_idx = splitobs(
        shuffleobs(collect(1:total_obs)), train_test_split
    )
    # Keys present in data => [:observed_data, :observed_mask, :observed_tp,
    #                          :tp_to_predict, :mask_predicted_data,
    #                          :data_to_predict]
    train_data = []
    test_data = []
    for key in [:observed_data, :observed_mask, :mask_predicted_data, :data_to_predict]
        push!(train_data, data[key][:, :, train_idx])
        push!(test_data, data[key][:, :, test_idx])
    end
    for key in [:observed_tp, :tp_to_predict]
        push!(train_data, data[key][:, train_idx])
        push!(test_data, data[key][:, test_idx])        
    end
    return (DataLoader(device_func.(train_data)..., batchsize = batchsize, shuffle = true),
            DataLoader(device_func.(test_data)..., batchsize = batchsize, shuffle = true))
end