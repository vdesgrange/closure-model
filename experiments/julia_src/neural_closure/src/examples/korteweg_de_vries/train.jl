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
    noise = 0.05; # noise
    batch_size = 16;
    sol = Rodas4P();
    xₙ = 64;
    xₘₐₓ = 8 * pi;
    Δx = xₘₐₓ / xₙ;
    snap_kwargs = (; Δx);

    opt = OptimizationOptimisers.ADAMW(lr, (0.9, 0.999), reg);
    dataset = Generator.read_dataset("./dataset/kdv_high_dim_m25_t10_128_x8pi_64_up2.jld2")["training_set"];
    model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
    K, p, _ = KdVCNN.training(model, epochs, dataset, opt, batch_size, ratio, noise, sol, snap_kwargs);
  
    @save "./models/kdv_high_dim_m25_t10_128_x30_64_up2_traj.bson" K p
  
    return K, p
  end

  K, p = main()
