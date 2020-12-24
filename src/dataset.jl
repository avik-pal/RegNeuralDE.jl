function load_mnist(batchsize::Int, transform = cpu)
    # Use MLDataUtils LabelEnc for natural onehot conversion
    onehot(labels_raw) =
        convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:9)))
    # Load MNIST
    imgs, labels_raw = MNIST.traindata()
    imgs = permutedims(imgs, (2, 1, 3))
    # Process images into (H,W,C,BS) batches
    x_train_data = Float32.(reshape(imgs, 28, 28, 1, size(imgs, 3)))
    y_train_data = onehot(labels_raw)
    # Load MNIST Test
    imgs, labels_raw = MNIST.testdata()
    imgs = permutedims(imgs, (2, 1, 3))
    # Process images into (H,W,C,BS) batches
    x_test_data = Float32.(reshape(imgs, 28, 28, 1, size(imgs, 3)))
    y_test_data = onehot(labels_raw)
    return (
        # Use Flux's DataLoader to automatically minibatch and shuffle the data
        DataLoader(
            transform.((x_train_data, y_train_data));
            batchsize = batchsize,
            shuffle = true,
        ),
        # Don't shuffle the test data
        DataLoader(
            transform.((x_test_data, y_test_data));
            batchsize = batchsize,
            shuffle = false,
        ),
    )
end

function load_miniboone(
    batchsize::Int,
    path::String,
    train_test_split::Float64 = 0.8,
    transform = cpu,
)
    data = Float32.(npzread(path)')
    total_obs = size(data, 2)
    train_data, test_data = splitobs(shuffleobs(data), train_test_split)

    return (
        DataLoader(transform.(train_data), batchsize = batchsize, shuffle = true),
        DataLoader(transform.(test_data), batchsize = batchsize, shuffle = false),
    )
end


function load_physionet(
    batchsize::Int,
    path::String,
    train_test_split::Float64 = 0.8,
    transform = cpu,
)
    data = BSON.load(path)[:dataset]
    for (key, value) in data
        ndims(value) == 1 && continue
        data[key] = permutedims(value, (3, 2, 1))
    end

    total_obs = size(data["observed_data"])[end]
    train_idx, test_idx = splitobs(shuffleobs(collect(1:total_obs)), train_test_split)
    train_data = []
    test_data = []
    for key in ["observed_data", "observed_mask", "data_to_predict", "mask_predicted_data"]
        push!(train_data, data[key][:, :, train_idx])
        push!(test_data, data[key][:, :, test_idx])
    end
    for key in ["observed_tp", "tp_to_predict"]
        push!(train_data, repeat(reshape(data[key], 1, :, 1), 1, 1, length(train_idx))),
        push!(test_data, repeat(reshape(data[key], 1, :, 1), 1, 1, length(test_idx)))
    end
    return (
        DataLoader(
            transform.(train_data)...,
            batchsize = batchsize,
            shuffle = true,
            partial = false,
        ),
        DataLoader(
            transform.(test_data)...,
            batchsize = batchsize,
            shuffle = true,
            partial = false,
        ),
    )
end


function load_spiral2d(
    batchsize::Int,
    transform = cpu;
    nspiral = 1000,
    ntotal = 500,
    nsample = 100,
    start = 0.0,
    stop = 1.0,
    noise_std = 0.1,
    a = 0.0,
    b = 1.0,
)
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
    for _ = 1:nspiral
        t₀ = rand(1:(ntotal-2*nsample)) + nsample - 1

        orig_traj = rand() > 0.5 ? orig_traj_cc : orig_traj_cw
        push!(original_trajectories, copy(orig_traj))

        samp_traj = copy(orig_traj[:, t₀:t₀+nsample-1, :])
        samp_traj .+= randn(size(samp_traj)) .* noise_std
        push!(sample_trajectories, samp_traj)
    end

    original_trajectories = Float32.(cat(original_trajectories..., dims = 3))
    original_tp = Float32.(reshape(repeat(collect(orig_ts), nspiral), :, nspiral))
    sampled_trajectories = Float32.(cat(sample_trajectories..., dims = 3))
    sampled_tp = Float32.(reshape(repeat(collect(samp_ts), nspiral), :, nspiral))

    return (
        DataLoader(
            transform.((sampled_trajectories, sampled_tp)),
            batchsize = batchsize,
            shuffle = true,
        ),
        DataLoader(
            transform.((original_trajectories, original_tp)),
            batchsize = batchsize,
            shuffle = true,
        ),
    )
end


function load_multimodel_gaussian(
    batchsize,
    transform = cpu,
    train_test_split = 0.8;
    nsamples = 1000,
    ngaussians = 6,
    dim = 2,
    radius = 5.0f0,
    σ = 0.1f0,
    noise = 0.3f0,
)
    samples_per_gaussian = nsamples ÷ ngaussians
    μ = zeros(Float32, dim)
    θ = 0
    X = Array{Float32}(undef, 2, samples_per_gaussian * ngaussians)
    for i = 1:ngaussians
        θ += Float32(2π / ngaussians)
        μ[1] = cos(θ) * radius
        μ[2] = sin(θ) * radius

        dist = MvNormal(μ, σ)
        samples = rand(dist, samples_per_gaussian)
        noise_vec = Float32.(randn(2, samples_per_gaussian)) .* noise
        X[:, (i-1)*samples_per_gaussian+1:i*samples_per_gaussian] = samples + noise_vec
    end

    X_train, X_test = splitobs(shuffleobs(X), train_test_split)

    return (
        DataLoader(transform.(X_train), batchsize = batchsize, shuffle = true),
        DataLoader(transform.(X_test), batchsize = batchsize, shuffle = false),
    )
end
