function load_mnist(batchsize::Int, transform = cpu)
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
        DataLoader(transform.(collect.((x_train_data, y_train_data)));
                   batchsize = batchsize, shuffle = true),
        # Don't shuffle the test data
        DataLoader(transform.(collect.((x_test_data, y_test_data)))...;
                   batchsize = batchsize, shuffle = false)
    )
end

function load_miniboone(batchsize::Int, path::String, train_test_split::Float64 = 0.8,
                        transform = cpu)
    data = Float32.(npzread(path)')
    total_obs = size(data, 2)
    train_data, test_data = splitobs(
        shuffleobs(data), train_test_split
    )

    return (DataLoader(transform.(train_data), batchsize = batchsize, shuffle = true),
            DataLoader(transform.(test_data), batchsize = batchsize, shuffle = false))
end


function load_physionet(batchsize::Int, path::String, train_test_split::Float64 = 0.8,
                        transform = cpu)
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
    return (DataLoader(transform.(train_data)..., batchsize = batchsize, shuffle = true),
            DataLoader(transform.(test_data)..., batchsize = batchsize, shuffle = true))
end


function load_spiral2d(batchsize::Int, transform = cpu; nspiral = 1000,
                       ntotal = 500, nsample = 100, start = 0.0,
                       stop = 1.0, noise_std = 0.1, a = 0.0, b = 1.0)
    # Toy Spiral 2D dataset for testing regularization on time series
    # extrapolation problem
    # A 2D spiral is parameterized by `r = a + b * theta`
    orig_ts = range(start, stop, length = ntotal)
    samp_ts = orig_ts[1:nsample]

    # clockwise and counter clockwissse spirals in observation space
    zs_cw = stop .+ 1.0f0 .- orig_ts
    rs_cw = a .+ b .* 50.0f0 ./ zs_cw
    xs = reshape(rs_cw .* cos.(zs_cw) .- 5.0f0, 1, :)
    ys = reshape(rs_cw .* sin.(zs_cw), 1, :)
    orig_traj_cw = reshape(cat(xs, ys, dims = 1), 2, :, 1)

    zs_cc = orig_ts
    rw_cc = a .+ b .* zs_cc
    xs = reshape(rw_cc .* cos.(zs_cc) .+ 5.0f0, 1, :)
    ys = reshape(rw_cc .* sin.(zs_cc), 1, :)
    orig_traj_cc = reshape(cat(xs, ys, dims = 1), 2, :, 1)

    sample_trajectories = []
    original_trajectories = []
    for _ in 1:nspiral
        t₀ = rand(1:(ntotal - 2 * nsample)) + nsample - 1

        orig_traj = rand() > 0.5 ? orig_traj_cc : orig_traj_cw
        push!(original_trajectories, copy(orig_traj))

        samp_traj = copy(orig_traj[:, t₀:t₀ + nsample - 1, :])
        samp_traj .+= randn(size(samp_traj)) .* noise_std
        push!(sample_trajectories, samp_traj)
    end
    
    original_trajectories = Float32.(cat(original_trajectories..., dims = 3))
    original_tp = Float32.(reshape(repeat(collect(orig_ts), nspiral), :, nspiral))
    sampled_trajectories = Float32.(cat(sample_trajectories..., dims = 3))
    sampled_tp = Float32.(reshape(repeat(collect(samp_ts), nspiral), :, nspiral))

    return (DataLoader(transform.((sampled_trajectories, sampled_tp)),
                       batchsize = batchsize, shuffle = true),
            DataLoader(transform.((original_trajectories, original_tp)),
                       batchsize = batchsize, shuffle = true))
end