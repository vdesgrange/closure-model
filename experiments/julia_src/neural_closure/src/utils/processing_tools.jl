module ProcessingTools

using CUDA
using Flux
using MLUtils
using Statistics
using IterTools: ncycle

add_dim(x::Array{Float64, 1}) = reshape(x, (size(x)[1], 1, 1))
add_dim(x::Array{Float64}) = reshape(x, (size(x)[1], 1,  size(x)[2:end]...))

perm_dim(x::Vector{Any}, x_n::Int64, t_n::Int64) = permutedims(reshape(hcat(x...), x_n, t_n, :), (1, 3, 2));


"""
  downsampling(u, d)

Downsample by d a matrix u. Compute average over cell of dimension d.
"""
function downsampling(u, d)
  (Nₓ, Nₜ) = size(u);
  Dₜ = Int64(Nₜ / d);
  u₁ = u[:, 1:d:end]; # Take t value every d steps
  Dₓ = Int64(Nₓ / d);
  u₂ = sum(reshape(u₁, d, Dₓ, Dₜ), dims=1);
  u₃   = reshape(u₂, Dₓ, Dₜ) ./ d;
  return u₃
end

function downsampling_x(u, d)
  (Nₓ, Nₜ) = size(u);
  Dₓ = Int64(Nₓ / d);
  u₂ = sum(reshape(u, d, Dₓ, Nₜ), dims=1);
  u₃   = reshape(u₂, Dₓ, Nₜ) ./ d;
  return u₃
end

"""
  process_dataset(dataset, keep_high_dim=true)

Process snapshot dataset of solution to ODE. 
Re-organize into 3 matrices: t values, initial values (x-snapshot), solution u (x-snapshot-t)
"""
function process_dataset(dataset, keep_high_dim=true)
  n = size(dataset, 1)

  # todo - split between training and validation data
  init_set = [];
  true_set = [];
  t = [];
  for i in range(1, size(dataset, 1), step=1)
    if keep_high_dim
      t, u, _, _ = dataset[i];
    else
      t, u = dataset[i];
    end

    push!(init_set, copy(u[:, 1]));
    push!(true_set, copy(u));
  end

  t_n = size(t, 1)
  x_n = size(init_set[1], 1)

  return t, hcat(init_set...), permutedims(reshape(hcat(true_set...), x_n, t_n, :), (1, 3, 2));
end


"""
  process_closure_dataset(dataset)

Process snapshot dataset of solution and ROM to ODE.
"""
function process_closure_dataset(dataset)
  n = size(dataset, 1);
  init_set = [];
  rom_set = [];
  true_set = [];
  t = [];

  for i in range(1, size(dataset, 1), step=1)
    t, ū, _, u, _, _ = dataset[i];

    push!(init_set, copy(ū[:, 1])); # must test with rom and fom for initial conditions
    push!(rom_set, copy(ū));
    push!(true_set, copy(u));
  end

  t_n = size(t, 1);
  x_n = size(init_set[1], 1);

  return t, hcat(init_set...), perm_dim(rom_set, x_n, t_n), perm_dim(true_set, x_n, t_n);
end

"""
  get_data_loader(dataset, batch_size, ratio)

Split dataset into training and validation set.
"""
function get_data_loader(dataset, batch_size, ratio, split_axis=true, cuda=true)
  if cuda && CUDA.has_cuda()
    CUDA.allowscalar(false)
    device = Flux.gpu
  else
    device = Flux.cpu
  end

  t, init_set, true_set = ProcessingTools.process_dataset(dataset, false);

  # True solution
  if (split_axis)
    t_train, t_val = splitobs(t, at = ratio);
    train_set, val_set = splitobs(true_set, at = ratio);
  else
    t_train, t_val = copy(t), copy(t);
    switch_true_set = permutedims(true_set, (1, 3, 2));
    train_set, val_set = splitobs(switch_true_set, at = ratio);
    train_set = permutedims(train_set, (1, 3, 2));
    val_set = permutedims(val_set, (1, 3, 2));
  end

  # Initial condition
  init_train = copy(train_set[:, :, 1]);
  init_val = copy(val_set[:, :, 1]);

  train_set = permutedims(train_set, (1, 3, 2));
  val_set = permutedims(val_set, (1, 3, 2));

  # Set size
  n_train = size(train_set, 3)
  n_val = size(val_set, 3)

  # Set data loader
  train_data = (init_train |> device, train_set |> device, collect(ncycle([collect(t_train)], n_train)))
  val_data = (init_val |> device, val_set |> device,  collect(ncycle([collect(t_val)], n_val)))

  train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true);
  val_loader = DataLoader(val_data, batchsize=n_val, shuffle=false);

  return (train_loader, val_loader)
end

"""
  get_data_loader_cnn(dataset, batch_size, ratio, split_axis, cuda)

Split dataset into training and validation set.
"""
function get_data_loader_cnn(dataset, batch_size, ratio, split_axis=true, cuda=true)
  if cuda && CUDA.has_cuda()
    CUDA.allowscalar(false)
    device = Flux.gpu
  else
    device = Flux.cpu
  end

  t, init_set, true_set = ProcessingTools.process_dataset(dataset, false);

  # True solution
  if (split_axis)
    t_train, t_val = splitobs(t, at = ratio);
    train_set, val_set = splitobs(true_set, at = ratio);
  else
    t_train, t_val = copy(t), copy(t);
    switch_true_set = permutedims(true_set, (1, 3, 2));
    train_set, val_set = splitobs(switch_true_set, at = ratio);
    train_set = permutedims(train_set, (1, 3, 2));
    val_set = permutedims(val_set, (1, 3, 2));
  end

  # Initial condition
  init_train = add_dim(copy(train_set[:, :, 1]));
  init_val = add_dim(copy(val_set[:, :, 1]));

  train_set = permutedims(train_set, (1, 3, 2));
  val_set = permutedims(val_set, (1, 3, 2));

  # Set size
  n_train = size(train_set, 3)
  n_val = size(val_set, 3)

  # Set data loader
  train_data = (init_train |> device, train_set |> device, collect(ncycle([collect(t_train)], n_train)))
  val_data = (init_val |> device, val_set |> device,  collect(ncycle([collect(t_val)], n_val)))

  train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true);
  val_loader = DataLoader(val_data, batchsize=n_val, shuffle=false);

  return (train_loader, val_loader)
end

function get_data_loader_rom(dataset, batch_size, ratio, split_axis=true, cuda=true)
  if cuda && CUDA.has_cuda()
    CUDA.allowscalar(false)
    device = Flux.gpu
  else
    device = Flux.cpu
  end

  t, init_set, rom_set, true_set = ProcessingTools.process_closure_dataset(dataset);

  # True solution
  if (split_axis)
    t_train, t_val = splitobs(t, at = ratio);
    train_rom_set, val_rom_set = splitobs(rom_set, at = ratio);
    train_set, val_set = splitobs(true_set, at = ratio);
  else
    t_train, t_val = copy(t), copy(t);

    switch_true_set = permutedims(true_set, (1, 3, 2));
    switch_rom_set = permutedims(rom_set, (1, 3, 2));

    train_set, val_set = splitobs(switch_true_set, at = ratio);
    train_set = permutedims(train_set, (1, 3, 2));
    val_set = permutedims(val_set, (1, 3, 2));

    train_rom_set, val_rom_set = splitobs(switch_rom_set, at = ratio);
    train_rom_set = permutedims(train_rom_set, (1, 3, 2));
    val_rom_set = permutedims(val_rom_set, (1, 3, 2));
  end

  # Initial condition
  init_train = add_dim(copy(train_set[:, :, 1]));
  init_val = add_dim(copy(val_set[:, :, 1]));

  train_rom_set = permutedims(train_rom_set, (1, 3, 2));
  val_rom_set = permutedims(val_rom_set, (1, 3, 2));

  train_set = permutedims(train_set, (1, 3, 2));
  val_set = permutedims(val_set, (1, 3, 2));

  # Set size
  n_train = size(train_set, 3)
  n_val = size(val_set, 3)

  # Set data loader
  train_data = (init_train |> device, train_rom_set, train_set |> device, collect(ncycle([collect(t_train)], n_train)))
  val_data = (init_val |> device, val_rom_set, val_set |> device,  collect(ncycle([collect(t_val)], n_val)))

  train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true);
  val_loader = DataLoader(val_data, batchsize=n_val, shuffle=false);

  return (train_loader, val_loader)
end

end
