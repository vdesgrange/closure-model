using BSON: @save
using Flux
using OrdinaryDiffEq: Tsit5
using OptimizationOptimisers

include("../../neural_ode/models.jl");
include("../../utils/generators.jl");
include("./kdv_cnn.jl");

function main()
    epochs = 100; # Iterations
    ratio = 0.75; # train/val ratio
    lr = 0.003; # learning rate
    reg = 1e-8; # weigh decay (L2 reg)
    noise = 0.; # noise
    batch_size = 16;
    sol = Tsit5();
    xₙ = 64;
    snap_kwargs = (; xₙ);

    opt = OptimizationOptimisers.ADAMW(lr, (0.9, 0.999), reg);
    dataset = Generator.read_dataset("./dataset/kdv_high_dim_m25_t0001_128_x1_64.jld2")["training_set"];
    model = Models.CNN3(9, [3, 4, 8, 8, 4, 2, 1]);
    K, p, _ = training(model, epochs, dataset, opt, batch_size, ratio, noise, Tsit5(), snap_kwargs);
  
    @save "./models/kdv_high_dim_m25_t0001_128_x1_64.bson" K p
  
    return K, p
  end

  K, p = main()