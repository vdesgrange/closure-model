module ProcessingTools

using CUDA
using Flux
using MLUtils
using Statistics
using IterTools: ncycle

function downsampling(u, d)
  """
    downsampling(u, d)

  Downsample by d a matrix u. Compute average over cell of dimension d.
  """
  n, m = floor.(Int, size(u) ./ d)
  d_u = zeros(n, m)

  for i in range(0, n - 1, step=1)
    for j in range(0, m - 1, step=1)
      d_u[i+1, j+1] = mean(u[i*d + 1:(i + 1)*d, j*d + 1:(j + 1)*d])
    end
  end

  return d_u
end

function process_dataset(dataset, keep_high_dim=true)
  """
    process_dataset(dataset, keep_high_dim=true)

  Process snapshot dataset of solution to ODE. 
  Re-organize into 3 matrices: t values, initial values (x-snapshot), solution u (x-snapshot-t)
  """
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

function get_data_loader(dataset, batch_size, ratio, split_axis=true, cuda=false)
  """
    get_data_loader(dataset, batch_size, ratio)

  Split dataset into training and validation set.
  """
  if cuda && CUDA.has_cuda()
    CUDA.allowscalar(true)
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

end
