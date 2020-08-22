function load_mnist(batchsize::Int)
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
        DataLoader(cpu.(collect.([x_train_data, y_train_data]))...; batchsize = batchsize,
                   shuffle = true),
        # Don't shuffle the test data
        DataLoader(cpu.(collect.([x_test_data, y_test_data]))...; batchsize = batchsize,
                   shuffle = false)
    )
end
