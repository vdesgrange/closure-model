using Distributed

pids = addprocs(10; exeflags=`--project=$(Base.active_project())`);

@everywhere using Distributed
@everywhere using Hyperopt
@everywhere using Hyperopt: @phyperopt
@everywhere using Flux
@everywhere using BSON: @save
@everywhere using FileIO
@everywhere using JLD2
@everywhere include("../../utils/generators.jl");
@everywhere include("../../neural_ode/models.jl");
@everywhere include("./burgers_direct.jl");
@everywhere include("./burgers_cnn.jl");


@everywhere function f_fnn(i, all)
    la = all[1]
    ne = all[2]
    x_n = 64; # Discretization
    epochs = 100; # Iterations
    ratio = 0.75; # train/val ratio
    lr = 0.03; # learning rate
    r = 1e-07; # weigh decay (L2 reg)
    n = 0.05; # noise
    b = 32;

    data = Generator.read_dataset("./dataset/high_dim_1k_set_j173.jld2")["training_set"];
    model = Models.FeedForwardNetwork(x_n, la, ne);
    opt = Flux.Optimiser(Flux.WeightDecay(r), Flux.ADAM(lr, (0.9, 0.999), 1.0e-8));
    K, p, _, l_val = BurgersDirect.training(model, epochs, data, opt, b, ratio, n);

    filename = "./models/feedforward_1k/tuning_burgers_05noise_reg_worker_" * string(myid()) * "_iter_" * string(i) * ".bson"
    @save filename K p

    return l_val
end

@everywhere function f_cnn(i, lr, batch, kernel, channels)
    epochs = 100; # Iterations
    ratio = 0.75; # train/val ratio
    r = 1e-07; # weigh decay (L2 reg)
    n = 0.05; # noise

    data = Generator.read_dataset("./dataset/viscous_burgers_high_dim_m10_256_j173.jld2")["training_set"];

    model = Models.CNN2(kernel, channels);
    opt = Flux.Optimiser(Flux.WeightDecay(r), Flux.ADAM(lr, (0.9, 0.999), 1.0e-8))
    K, p, _, l_val = BurgersCNN.training(model, epochs, data, opt, batch, ratio, n);

    filename = "./models/cnn_viscous_256/tuning_burgers_05noise_reg_worker_" * string(myid()) * "_iter_" * string(i) * ".bson"
    @save filename K p

    return l_val
end

lr = [0.01, 0.003, 0.001];
batch = [8, 16, 32];
kernel = [3, 5, 9];
channels = [
    [2, 2, 1],
    [2, 4, 4, 2, 1],
    [2, 4, 8, 8, 4, 2, 1],
];
# r = [1f-8, 1f-7, 1f-6, 1f-5, 1f-4, 1f-3, 1f-2, 1f-1],
# n = [.35, .3, .25, .2, .15, .1, .05, .01]
# la = [1, 2, 3, 4, 5];
# ne = [8, 16, 24, 32, 40, 48, 56, 64];
# n = [0.0]
# r = collect(LinRange(1f-12, 1f-1, 20))
# all_set = [];
# useless_set = [];
# for k in kernel
#     for c in channels
#         for b in batch
#             push!(all_set, (k, c, b))
#             push!(useless_set, 1)
#         end
#     end
# end

ho = @phyperopt for i = 15,
    sampler = LHSampler(),
        l = repeat(lr, 5),
        b = repeat(batch, 5),
        k = repeat(kernel, 5),
        c = repeat(channels, 5)
        # all = all_set,
        # useless = useless_set
    #print(i, "\t", all[1], "\t", all[2], "\t")
    print(i, "\t lr=", l, "\t batch=", b, "\t kernel=", k, "\t channels=", c, "\t")
    l = f_cnn(i, l, b, k, c);
    print(i, "\t lr=", l, "\t batch=", b, "\t kernel=", k, "\t channels=", c, "\t f_cnn = ", l, "    \t")
    @show l
end

JLD2.save("hyperopt_viscous_burgers_05noise.jld2", "ho", ho);

interrupt(pids);
