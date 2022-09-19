using BSON: @save
using Flux
using OrdinaryDiffEq: Tsit5
using OptimizationOptimisers

include("../../neural_ode/models.jl");
include("../../utils/generators.jl");
include("./burgers_cnn.jl");
include("./burgers_combined_optimizer.jl");

function main()
  x_n = 64; # Discretization
  epochs = 500; # Iterations
  ratio = 0.75; # train/val ratio
  lr = 0.003; # learning rate
  reg = 1e-7; # weigh decay (L2 reg)
  n = 0.05; # noise
  batch = 16; # Batch size

  opt = Flux.Optimiser(Flux.WeightDecay(reg), Flux.ADAM(lr, (0.9, 0.999), 1.0e-8))
  data = Generator.read_dataset("./dataset/inviscid_burgers_high_dim_m4_256_j173.jld2")["training_set"];
  model = Models.CNN2(9, [2, 4, 4, 2, 1]);
  K, p, _, _ = BurgersCNN.training(model, epochs, data, opt, batch, ratio, n);
  @save "./models/inviscid_burgers_high_dim_m4_256_model_j173.bson" K p

  return K, p
end

function main2()
  x_n = 64; # Discretization
  epochs = 500; # Iterations
  ratio = 0.75; # train/val ratio
  lr = 0.003; # learning rate
  reg = 1e-7; # weigh decay (L2 reg)
  n = 0.05; # noise
  batch = 16; # Batch size

  opt = Flux.Optimiser(Flux.WeightDecay(reg), Flux.ADAM(lr, (0.9, 0.999), 1.0e-8))
  data = Generator.read_dataset("./dataset/inviscid_burgers_high_dim_m4_256_j173.jld2")["training_set"];
  model = Models.CNN2(9, [2, 2, 1]);
  K, p, _, _ = BurgersCNN.training(model, epochs, data, opt, batch, ratio, n);
  @save "./models/inviscid_burgers_high_dim_m4_256_model_2_j173.bson" K p

  return K, p
end

function main3()
  x_n = 64; # Discretization
  epochs = 500; # Iterations
  ratio = 0.75; # train/val ratio
  lr = 0.003; # learning rate
  reg = 1e-7; # weigh decay (L2 reg)
  n = 0.05; # noise
  batch = 16; # Batch size

  opt = Flux.Optimiser(Flux.WeightDecay(reg), Flux.ADAM(lr, (0.9, 0.999), 1.0e-8))
  data = Generator.read_dataset("./dataset/inviscid_burgers_gauss_256_j173.jld2")["training_set"];
  model = Models.CNN2(9, [2, 2, 1]);
  K, p, _, _ = BurgersCNN.training(model, epochs, data, opt, batch, ratio, n);
  @save "./models/inviscid_burgers_gauss_256_model_j173.bson" K p

  return K, p
end

function main4()
  x_n = 64; # Discretization
  epochs = 500; # Iterations
  ratio = 0.75; # train/val ratio
  lr = 0.001; # learning rate
  reg = 1e-7; # weigh decay (L2 reg)
  n = 0.05; # noise
  batch = 16; # Batch size

  opt = Flux.Optimiser(Flux.WeightDecay(reg), Flux.ADAM(lr, (0.9, 0.999), 1.0e-8))
  data = Generator.read_dataset("./dataset/viscous_burgers_high_dim_m10_256_j173.jld2")["training_set"];
  model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
  K, p, _, _ = BurgersCNN.training(model, epochs, data, opt, batch, ratio, n);
  @save "./models/viscous_burgers_high_dim_m10_256_500epoch_model_j173.bson" K p

  return K, p
end

function main5()
  x_n = 64; # Discretization
  epochs = 500; # Iterations
  ratio = 0.75; # train/val ratio
  lr = 0.001; # learning rate
  reg = 1e-7; # weigh decay (L2 reg)
  n = 0.05; # noise
  batch = 16;

  opt = OptimizationOptimisers.ADAMW(lr, (0.9, 0.999), reg);
  data = Generator.read_dataset("./dataset/viscous_burgers_high_dim_m10_256_j173.jld2")["training_set"];
  model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
  K, p, _ = BurgersCombinedCNN.training(model, epochs, data, opt, batch, ratio, n, Tsit5());
  @save "./models/viscous_burgers_high_dim_m10_256_500epoch_model2_j173.bson" K p

  return K, p
end

function main6()
  x_n = 64; # Discretization
  epochs = 50; # Iterations
  ratio = 0.75; # train/val ratio
  lr = 0.003; # learning rate
  reg = 1e-7; # weigh decay (L2 reg)
  n = 0.1; # noise
  batch = 16;

  opt = OptimizationOptimisers.ADAMW(lr, (0.9, 0.999), reg);
  data = Generator.read_dataset("./dataset/viscous_burgers_high_dim_m10_256_j173.jld2")["training_set"];
  model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
  K, p, _ = BurgersCombinedCNN.training(model, epochs, data, opt, batch, ratio, n, Tsit5());
  @save "./models/inviscid_burgers_high_dim_m10_256_500epoch_model2_j173.bson" K p

  return K, p
end
