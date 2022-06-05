module ProcessingTools

using Statistics

function downsampling(u, d)
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

    push!(init_set, copy(u[:, 1])); # make copies ?
    push!(true_set, copy(u)); # make copies ?
  end

  t_n = size(t, 1)
  x_n = size(init_set[1], 1)

  return t, hcat(init_set...), permutedims(reshape(hcat(true_set...), x_n, t_n, :), (1, 3, 2));
end

end
