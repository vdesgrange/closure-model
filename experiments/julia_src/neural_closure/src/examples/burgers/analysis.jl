module BurgersAnalysis

using Plots
using OrdinaryDiffEq
using DiffEqFlux
using MLUtils

include("../../utils/generators.jl")
include("../../utils/processing_tools.jl")
include("../../utils/graphic_tools.jl")

function check_result(nn, res, typ)
    t, u0, u = Generator.get_burgers_batch(0.5, 0., pi, 0., 64, 64, 0.03, typ);
    prob_neuralode = DiffEqFlux.NeuralODE(nn, (t[1], t[end]), AutoTsit5(Rosenbrock23()), saveat=t);
    u_pred = prob_neuralode(u0, res);

    plot(
        GraphicTools.show_state(u, ""),
        GraphicTools.show_state(hcat(u_pred.u...), ""),
        GraphicTools.show_err(hcat(u_pred.u...), u, "");
        layout = (1, 3),
    )
    savefig("burgers_direct_results.png");
end

function inter_extra_fit(K, p, n, t_max, t_min, x_max, x_min, t_n, x_n, nu, typ)
    """
        inter_extra_fit(K, p, n, t_max, t_min, x_max, x_min, t_n, x_n, nu, typ)

    Generate unique snapshot of solutions to Burgers equation. Depending on the parameter, it helps:
    analyse interpolation (reconstruction) accuracy by solving ODE on range of training
    analyse extrapolation (prediction) accuracy by solving ODE on range unknown.

    # Arguments
    - `K` : neural operator
    - `p` : neural network weights/bias
    - `n` : number of snapshots generated
    - `t_max` : maximum t value
    - `t_min` : minimum t value
    - `x_max` : maximum x value
    - `x_min` : minimum x value
    """
    mse_tot = 0
    l1_tot = 0

    for i in 1:n
        t, u0, u_true = Generator.get_burgers_batch(t_max, t_min, x_max, x_min, t_n, x_n, nu, typ);
        prob_neuralode = DiffEqFlux.NeuralODE(K, (t[1], t[end]), AutoTsit5(Rosenbrock23()), saveat=t);
        u_pred = prob_neuralode(u0, p);
        mse_tot += Flux.mse(u_pred, u_true)
        l1_tot += Flux.mae(u_pred, u_true)
    end

    return mse_tot / n, l1_tot / n
end

function validation_fit(K, p, data, ratio)
    t, init, u = ProcessingTools.process_dataset(data, true);
    t_train, t_val = splitobs(t, at = ratio);
    train_set, val_set = splitobs(u, at = ratio);

    n = size(init, 2)
    mse_tot = 0
    l1_tot = 0
    mse_train = 0
    l1_train = 0
    mse_val = 0
    l1_val = 0

    for i in 1:n
        prob_neuralode = DiffEqFlux.NeuralODE(K, (t[1], t[end]), Tsit5(), saveat=t);
        u0 = init[:, i]
        u_pred = prob_neuralode(u0, p);
        mse_tot += Flux.mse(u_pred, u[:, i, :])
        l1_tot += Flux.mae(u_pred, u[:, i, :])

        prob_neuralode = DiffEqFlux.NeuralODE(K, (t_train[1], t_train[end]), Tsit5(), saveat=t_train);
        u0 = init[:, i]
        u_pred = prob_neuralode(u0, p);
        mse_train += Flux.mse(u_pred, train_set[:, i, :])
        l1_train += Flux.mae(u_pred, train_set[:, i, :])

        prob_neuralode = DiffEqFlux.NeuralODE(K, (t_val[1], t_val[end]), Tsit5(), saveat=t_val);
        u0 = copy(val_set[:, i, 1]);
        u_pred = prob_neuralode(u0, p);
        mse_val += Flux.mse(u_pred, val_set[:, i, :])
        l1_val += Flux.mae(u_pred, val_set[:, i, :])
    end

    return mse_tot / n, l1_tot / n, mse_train / n, l1_train / n, mse_val / n, l1_val / n
end

function generate_variational_nu_set()
end

# test_simulation(K, 100, 3)

end
