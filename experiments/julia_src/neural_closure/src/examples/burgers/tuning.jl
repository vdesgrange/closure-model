using Distributed

pids = addprocs(2; exeflags=`--project=$(Base.active_project())`);

@everywhere using Distributed
@everywhere using Hyperopt
@everywhere using Hyperopt: @phyperopt
@everywhere using BSON: @save
@everywhere using FileIO
@everywhere using JLD2
@everywhere include("../../utils/generators.jl")
@everywhere include("../../neural_ode/models.jl")
@everywhere include("./burgers_direct.jl")


@everywhere function f_autoencoder(i, lr, b, r, n)
    x_n = 64;
    epochs = 100;
    ratio = 0.7;

    data = Generator.read_dataset("./dataset/burgers_high_dim_training_set.jld2")["training_set"];
    model = Models.BasicAutoEncoder(x_n);
    K, p, l_train, _ = BurgersDirect.training(model, epochs, data, b, ratio, lr, n, r, false);

    filename = "./models/tuning_burgers_basicautoencoder_2_worker_" * string(myid()) * "_iter_" * string(i) * ".bson"
    @save filename K p

    return l_train
end

@everywhere function f_fnn(i, all)
    la = all[1]
    ne = all[2]
    x_n = 64; # Discretization
    epochs = 100; # Iterations
    ratio = 0.7; # train/val ratio
    lr = 0.03; # learning rate
    r = 1e-07; # weigh decay (L2 reg)
    n = 0.05; # noise
    b = 32;

    # data = Generator.read_dataset("./dataset/burgers_high_dim_nu_variational_dataset.jld2")["training_set"];
    data = Generator.read_dataset("./dataset/burgers_high_dim_training_set.jld2")["training_set"];
    model = Models.FeedForwardNetwork(x_n, la, ne);
    K, p, l_train, _ = BurgersDirect.training(model, epochs, data, b, ratio, lr, n, r, false);

    filename = "./models/feedforward2/tuning_burgers_fnn_05noise_worker_" * string(myid()) * "_iter_" * string(i) * ".bson"
    @save filename K p

    return l_train
end

# lr = [0.1, 0.03, 0.01, 0.003],
# b = [128, 64, 32, 16],
# r = [1f-8, 1f-7, 1f-6, 1f-5, 1f-4, 1f-3, 1f-2, 1f-1],
# n = [.35, .3, .25, .2, .15, .1, .05, .01]
la = [1, 2, 3, 4, 5];
ne = [8, 16, 24, 32, 40];
all_set = [];
useless_set = [];
for k in la
    for l in ne
        push!(all_set, (k, l))
        push!(useless_set, 1)
    end
end

# Random.seed!(0)
ho = @phyperopt for i = 25,
    sampler = LHSampler(),
        all = all_set,
        useless = useless_set
    print(i, "\t", all[1], "\t", all[2], "\t")
    l = f_fnn(i, all);
    print(i, "\t", all[1], "\t", all[2], "\t f_autoencoder(la, ne) = ", l, "    \t")
    @show l
end

JLD2.save("hyperopt_result_fnn_05noise.jld2", "ho", ho);

interrupt(pids);
