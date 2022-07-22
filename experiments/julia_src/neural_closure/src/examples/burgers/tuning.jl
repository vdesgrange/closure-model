module HO

using Distributed
using Distributed
using Hyperopt
using Hyperopt: @phyperopt
using BSON: @save
include("../../utils/generators.jl")
include("../../neural_ode/models.jl")
include("./burgers_direct.jl")


function f_autoencoder(i, lr, b, r, n)
    x_n = 64;
    epochs = 100;
    ratio = 0.7;

    data = Generator.read_dataset("./dataset/burgers_high_dim_training_set.jld2")["training_set"];
    model = Models.BasicAutoEncoder(x_n);
    K, p, l_train, _ = BurgersDirect.training(model, epochs, data, b, lr, ratio, n, r, true);

    filename = "./models/tuning_burgers_basicautoencoder_" * string(i) * ".bson"
    @save filename K p

    return l_train
end


function tuning(n_iteration)
    # pids = addprocs(2; exeflags=`--project=$(Base.active_project())`);

    logs = [];
    ho = @hyperopt for i = n_iteration,
        sampler = Hyperopt.RandomSampler(),
            lr = LinRange(1f-4, 1f-1, 8),
            b = [128, 64, 32, 16],
            r = [1f-8, 1f-7, 1f-6, 1f-5, 1f-4, 1f-3, 1f-2, 1f-1],
            n = [.35, .3, .25, .2, .15, .1, .05, .01]
        l = f_autoencoder(i, lr, b, r, n);
        @show "\t", i, "\t", b, "\t", r, "\t", n, "   \t f_autoencoder(b, r, n) = ", l, "    \t"
        push!(logs, string(i, " ", b, " ", r, " ", n, "     f_autoencoder(b, r, n) = ", l, "  "));
    end

    for log in logs
        @show log
    end

    # interrupt(pids);

    return ho, logs
end

end

export HO
