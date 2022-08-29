using Flux

function CAE(x_n, l, k)
  encoder = []
  decoder = []

  for i in range(1, l, step=1)
    conv = Flux.Conv((k, 1), 1 => 1, tanh; stride = 1, pad = SamePad(), bias=true, init=Flux.glorot_uniform);
    pool = Flux.MeanPool((2, 1), pad = SamePad());
    push!(encoder, conv);
    push!(encoder, pool);

    upscale = Upsample(:nearest, scale = (2, 1))
    deconv = Flux.ConvTranspose((k, 1), 1 => 1, tanh; stride = 1, pad = SamePad(), bias=true, init=Flux.glorot_uniform);
    push!(decoder, upscale);
    push!(decoder, deconv);
  end

  return Flux.Chain(
    Flux.Dense(x_n => x_n, tanh; init=Flux.glorot_uniform, bias=true),
    encoder...,
    decoder...,
    Flux.Dense(x_n => x_n, identity; init=Flux.glorot_uniform, bias=true),
  )
end

function CAE2(k)
  encoder = [
    Flux.Conv((k, 1), 1 => 4, sigmoid; stride = 2, dilation = 1, pad = SamePad(), bias=true, init=Flux.glorot_uniform),
    # Flux.Conv((k, 1), 4 => 8, sigmoid; stride = 2, dilation = 1, pad = SamePad(), bias=true, init=Flux.glorot_uniform),
    # Flux.Conv((k, 1), 8 => 16, sigmoid; stride = 2, dilation = 1, pad = SamePad(), bias=true, init=Flux.glorot_uniform),
  ];

  decoder = [
    # Flux.ConvTranspose((k, 1), 16 => 8, sigmoid; stride = (2, 1), dilation = 1, pad = SamePad(), bias=true, init=Flux.glorot_uniform),
    # Flux.ConvTranspose((k, 1), 8 => 4, sigmoid; stride = (2, 1), dilation = 1, pad = SamePad(), bias=true, init=Flux.glorot_uniform),
    Flux.ConvTranspose((k, 1), 4 => 1, identity; stride = (2, 1), dilation = 1, pad = SamePad(), bias=true, init=Flux.glorot_uniform),
  ];

  return Flux.Chain(
    encoder...,
    # x -> reshape(x, :, size(x, 4)),
    # Flux.Dense(128 => 64, sigmoid; init=Flux.glorot_uniform, bias=true),
    # Flux.Dense(64 => 64, sigmoid; init=Flux.glorot_uniform, bias=true),
    # Flux.Dense(64 => 128, sigmoid; init=Flux.glorot_uniform, bias=true),
    # x -> reshape(x, 8, 1, 16, size(x, 2)),
    decoder...,
  )
end

