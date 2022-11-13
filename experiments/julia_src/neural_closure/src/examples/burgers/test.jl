using Plots;

include("../../utils/generators.jl");
include("../../equations/burgers/burgers_gp.jl")
include("../../utils/graphic_tools.jl")
include("../../rom/pod.jl")

tₘₐₓ= 2.;
tₘᵢₙ = 0.;
xₘₐₓ = pi;
xₘᵢₙ = 0;
tₙ = 64;
xₙ = 64;
ν = 0.04;
x = LinRange(xₘᵢₙ, xₘₐₓ, xₙ);
t = LinRange(tₘᵢₙ, tₘₐₓ, tₙ);
Δx = (xₘₐₓ / (xₙ - 1));
Δt = (tₘₐₓ / (tₙ - 1));
m = 50;
snap_kwarg= repeat([(; tₘₐₓ, tₘᵢₙ, xₘₐₓ, xₘᵢₙ, tₙ, xₙ, ν, typ=2)], 256);
init_kwarg = repeat([(; mu=10)], 256);
# dataset = Generator.generate_closure_dataset(256, 16, "", snap_kwarg, init_kwarg);
# dataset = Generator.read_dataset("kdv_high_dim_m25_t10_128_x30_64_up8.jld2")["training_set"];

dataset = Generator.read_dataset("./dataset/viscous_burgers_high_dim_t2_64_xpi_64_nu0.04_typ2_m10_256_up16_j173.jld2")["training_set"];
for (i, data) in enumerate(dataset)
  t, u, _, _ = data;
  # display(GraphicTools.show_state(u, t, x, "", "t", "x"))
end

# === Test ===
begin
  tmp = []
  for (i, data) in enumerate(dataset)
    push!(tmp, data[2]);
  end
  u_ref = cat(tmp...; dims=3);
  # size(u_ref)

  # u_ref = dataset[1][2]
  umean = mean(reshape(Array(u_ref[:, :, 1]), xₙ, :); dims = 2)
  bas, _ = POD.generate_pod_svd_basis(reshape(Array(u_ref), xₙ, :), false);
  λ = bas.eigenvalues;
  POD.get_energy(λ, m)
  Φ = bas.modes[:, 1:m];

  t, u₀, u = Generator.get_burgers_batch(2., 0., pi, 0., 64, 64, 0.04, 2, (; m=5));

  display(GraphicTools.show_state(u, t, x, "", "t", "x"))
  # display(GraphicTools.show_state(Φ * Φ' * u, t, x, "", "t", "x"))
  # display(GraphicTools.show_state(umean .+ Φ * Φ' * (u .- umean), t, x, "", "t", "x"))
  gp = galerkin_projection(t, u, Φ, 0.04, Δx, Δt);
  display(GraphicTools.show_state(gp, t, x, "", "t", "x"))

  size(Φ)
  plot(x, Φ[:, 1:3])
  Plots.heatmap(Φ' * Φ; xmirror = true, yflip = true)
  Plots.heatmap(Φ * Φ'; xmirror = true, yflip = true)
  Plots.heatmap(I(64); xmirror = true, yflip = true)


  for (i, t) ∈ enumerate(t)
    pl = Plots.plot(; xlabel = "x")
    Plots.plot!(pl, x, u[:, i]; label = "Reference")
    Plots.plot!(pl, x, Φ * Φ' * u[:, i]; label = "POD")
    Plots.plot!(pl, x, gp[:, i]; label = "Galerkin Projection")
    display(pl)
    # sleep(0.05) # Time for plot pane to update
  end
end

# ==========

dataset = Generator.read_dataset("./dataset/viscous_burgers_high_dim_t2_64_xpi_64_nu0.04_typ2_m10_256_up16_j173.jld2")["training_set"];
Φ_dataset, train_dataset = dataset[1:128], dataset[129:end];

# === Get basis Φ ===
function get_Φ(Φ_dataset, m)
  tmp = [];
  for (i, data) in enumerate(Φ_dataset)
    push!(tmp, data[2]);
  end
  u_cat = Array(cat(tmp...; dims=3));
  xₙ = size(u_cat, 1)
  bas, _ = POD.generate_pod_svd_basis(reshape(u_cat, xₙ, :), false);
  λ = bas.eigenvalues;
  @show POD.get_energy(λ, m)

  Φ = bas.modes[:, 1:m];

  return Φ
end

Φ = get_Φ(Φ_dataset, m);

# === Get reference data-set ===

ref_set = [];
gp_set = [];
for (i, data) in enumerate(train_dataset)
  (t, u, snap_kwarg, init_kwarg) = data;
  
  # 0. FOM v(t) = Φ'u(t)
  v = Φ' * u; # m - by - x_n
  push!(ref_set, [t, v, snap_kwarg, init_kwarg])

  # 1. ROM vₜ = g(v) (Galerkin Projection)
  gp = galerkin_projection(t, u, Φ, ν, Δx, Δt);
  push!(gp_set, [t, gp, snap_kwarg, init_kwarg])
end

# ==== training ====

using Flux
using Optimization
using OptimizationOptimisers
using DiffEqSensitivity
using IterTools: ncycle

include("../../utils/processing_tools.jl")
include("../../neural_ode/models.jl")
include("../../neural_ode/regularization.jl")

global ep = 0;
global count = 0;
epochs = 100;
ratio = 0.75;
batch_size = 8;
lr = 1e-3;
reg = 1e-7;
noise = 0.05;
# model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
model = Flux.Chain(
  v -> vcat(v, v.^2),
  Flux.Dense(2m => m, tanh; init=Flux.glorot_uniform, bias=true),
  # Flux.Dense(m => m, tanh; init=Flux.glorot_uniform, bias=true),
  Flux.Dense(m => m, identity; init=Flux.glorot_uniform, bias=true),
)

function f(u, p, t)
  ν = p[1]
  Δx = p[2]
  u₋ = circshift(u, 1)
  u₊ = circshift(u, -1)
  a₊ = u₊ + u
  a₋ = u + u₋
  du = @. -((a₊ < 0) * u₊^2 + (a₊ > 0) * u^2 - (a₋ < 0) * u^2 - (a₋ > 0) * u₋^2) / Δx + ν * (u₋ - 2u + u₊) / Δx^2
  du
end


del_dim(x::Array{Float64, 4}) = reshape(x, (size(x)[1], size(x)[3], size(x)[4]))
del_dim(x::Array{Float64, 3}) = x

@info("Loading dataset") # train_dataset ou reference
(train_loader, val_loader) = ProcessingTools.get_data_loader_cnn(train_dataset, batch_size, ratio, false, false);

@info("Building model")
p, re = Flux.destructure(model);
fᵣₒₘ =  (v, p, t) -> re(p)(v)

function predict_neural_ode(θ, x, t)
  _prob = ODEProblem(fᵣₒₘ, x, extrema(t), θ, saveat=t);
  ȳ = Array(solve(_prob, Tsit5(), sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
  return permutedims(del_dim(ȳ), (1, 3, 2));
end

function loss_trajectory_fit(θ, x, u, t)
  xₙ, tₙ, bₙ = size(u);
  mₙ = size(Φ, 2);
  u = reshape(u, xₙ, :)
  x = reshape(x, xₙ, :)

  x̂ = Reg.gaussian_augment(Φ' * x, noise);
  x̂ = reshape(x̂, mₙ, 1, bₙ)

  v̂  = predict_neural_ode(θ, x̂, t[1]);

  v = Φ' * u; 
  v = reshape(v, mₙ, tₙ, :);

  l = Flux.mse(v̂, v)  + reg * sum(θ);
  return l;
end

function loss_derivative_fit(θ, x, u, t)
  xₙ, tₙ, bₙ = size(u);
  mₙ = size(Φ, 2);

  u = reshape(u, xₙ, :)
  v = Φ' * u; 
  v = reshape(v, mₙ, tₙ, :);

  dv = Φ' * f(u, (ν, Δx), t)
  dv = reshape(v, mₙ, tₙ, :);

  #v = reshape(v, mₙ, 1, tₙ * bₙ) # CNN
  dv̂ = fᵣₒₘ(v, θ, t);
  #dv̂ = reshape(dv̂, mₙ, tₙ, bₙ) # CNN

  l = Flux.mse(dv̂, dv) + reg * sum(θ);
  return l;
end

function val_loss(θ, x, u, t)
  x = reshape(x, size(x, 1), :);
  x = Φ' * x;
  x = reshape(x, size(x, 1), 1, :);

  v̂ = predict_neural_ode(θ, x, t[1]);

  u = reshape(u, size(u, 1), :)
  v = Φ' * u; 
  v = reshape(v, size(v̂, 1), size(v̂, 2), :);

  lt = Flux.mse(v̂, v);
  return lt;
end

function cb(θ, l)
  @show(l)
  global count += 1;

  iter = (train_loader.nobs / train_loader.batchsize);
  if (count % iter == 0)
      global ep += 1;
      global count = 0;
      ltraj = 0;

      for (x, y, t) in val_loader
          ltraj += val_loss(θ, x, y, t); 
      end
      ltraj /= (val_loader.nobs / val_loader.batchsize);

      @info("Epoch ", ep, ltraj);
  end
  return false
end


@info("Initiate training")
@info("ADAM Trajectory fit")
opt = OptimizationOptimisers.Adam(lr, (0.9, 0.999)); 
optf = OptimizationFunction((θ, p, x, y, t) -> loss_trajectory_fit(θ, x, y, t), Optimization.AutoZygote());
optprob = Optimization.OptimizationProblem(optf, p);
result_neuralode = Optimization.solve(optprob, opt, ncycle(train_loader, epochs), callback=cb);
K = (u, p, t) -> re(p)(u);
θ = result_neuralode.u;

v₀
re(θ)(v₀)
re(θ)(reshape(v₀, :, 1))

# === Check results ===
u = train_dataset[1][2];
u₀ = u[:, 1];
v = Φ' * u;
v₀ = v[:, 1];
û = gp_set[1][2]; # include Φ

_prob = ODEProblem(K, reshape(v₀, :, 1, 1), extrema(t), θ; saveat=t);
v̄ = solve(_prob, Tsit5());
# v̄ = Array(solve(_prob, Tsit5(), p=θ, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
size(v̄)
display(GraphicTools.show_state(Φ * v, t, x, "", "t", "x"))
display(GraphicTools.show_state(Φ * v̄[:, 1, 1, :], t, x, "", "t", "x"))
display(GraphicTools.show_state(Φ * v̂, t, x, "", "t", "x"))

for (i, t) ∈ enumerate(t)
  pl = Plots.plot(; xlabel = "x", ylim=extrema(v))
  Plots.plot!(pl, x, u[:, i]; label = "FOM - Model 0")
  Plots.plot!(pl, x, û[:, i]; label = "ROM - Model 1")
  Plots.plot!(pl, x, Φ * v̄[:, 1, 1, i]; label = "ROM - Model 2")
  display(pl)
  # sleep(0.05)
end

