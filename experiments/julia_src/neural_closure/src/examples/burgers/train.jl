include("../../neural_ode/models.jl");
include("../../utils/generators.jl");
include("./analysis.jl");

function main()
  x_n = 64; # Discretization
  epochs = 20; # Iterations
  ratio = 0.75; # train/val ratio
  lr = 0.03; # learning rate
  reg = 1e-7; # weigh decay (L2 reg)
  n = 0.05; # noise
  batch = 32; # Batch size
  hl = 3; # Hidden layers
  ne = 48;  # Neurons

  opt = Flux.Optimiser(Flux.WeightDecay(reg), Flux.ADAM(lr, (0.9, 0.999), 1.0e-8))

  data = Generator.read_dataset("./dataset/high_dim_1k_set_j173.jld2")["training_set"];
  model = Models.FeedForwardNetwork(x_n, hl, ne);
  K, p, _, _ = training(model, epochs, data, opt, batch, ratio, n, true);
  @save "./models/high_dim_1k_model_j173.bson" K p

  return K, p
end
