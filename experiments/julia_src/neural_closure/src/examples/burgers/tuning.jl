module HO

using Hyperopt
using Hyperopt: @hyperopt
using BSON: @save 

include("../../utils/generators.jl")
include("../../neural_ode/models.jl")
include("./burgers_direct.jl")

function f_autoencoder(i, b, r, n)
    x_n = 64;
    epochs = 10;
    data = Generator.read_dataset("./dataset/burgers_high_dim_training_set.jld2")["training_set"];
    model = Models.BasicAutoEncoder(x_n);
    K, p, l_train, _ = BurgersDirect.training(model, epochs, data, b, 0.5, n, r, true);
   
    filename = "./models/tuning_burgers_basicautoencoder_" * string(i) * ".bson"
    @save filename K p

    return l_train
end

function tuning(n_iteration)
    logs = [];
    ho = @hyperopt for i = n_iteration,
            sampler = Hyperopt.RandomSampler(),
            b = [128, 64, 32],
            r = [1f-8, 1f-6, 1f-5, 1f-4, 1f-3, 1f-2, 1f-1],
            n = [.3, .2, .15, .1, .05, .01]
        l = f_autoencoder(i, b, r, n);
        @show l
        push!(logs, string(i, "\t", b, "\t", r, "\t", n, "   \t f_autoencoder(b, r, n) = ", l, "    \t"));
    end

    for log in logs
        @show log
    end

    return logs
end

end
