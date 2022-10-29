using FFTW
using AbstractFFTs
using OrdinaryDiffEq
using Plots
using PyPlot
using LinearAlgebra
using SparseArrays
using Random
using Distributions
using JLD2

include("../../equations/initial_functions.jl")
include("../../utils/graphic_tools.jl")
include("../../utils/generators.jl")

function show_state(u, x, y, title, xlabel, ylabel)
  pyplot();

  xₙ, yₙ = size(x)[1], size(y)[1];
  xₘᵢₙ, xₘₐₓ = x[1], x[end];
  yₘᵢₙ, yₘₐₓ = y[1], y[end];

  xformatter = x -> string(round(x / xₙ * xₘₐₓ + xₘᵢₙ, digits=2));
  yformatter = y -> string(round(y / yₙ * yₘₐₓ + yₘᵢₙ, digits=2));

  pl = heatmap(u);
  heatmap!(pl,
      title = title,
      dpi=600,
      aspect_ratio = :equal,
      reuse=false,
      c=:dense,
      xlabel=xlabel,
      ylabel=ylabel,
      xticks=(1:7:size(x)[1], [xformatter(x) for x in 0:7:size(x)[1]]),
      yticks=(1:7:size(y)[1], [yformatter(y) for y in 0:7:size(y)[1]]),
  );

  return pl;
end

function random_init(t, x)
  u = zeros(Float64, size(t, 1), size(x, 1))
  d = Normal(0, .25)
  nu = rand(d, size(x, 1))
  u[1, :] .= sin.(x) + nu
  u[:, 1] .= 0
  u[:, end] .= 0

  return u
end


t_max = 10;  # 0.0001, 10
t_min = 0;
x_max = 8 * pi; # 1, 8 * pi,
x_min = 0;
x_n = 128;
t_n = 256;

# t = LinRange(0, 0.0002, 51)
t = LinRange(t_min, t_max, t_n);
x = LinRange(x_min, x_max, x_n);

Δx = (x_max - x_min) / x_n;
Δt = (t_max - t_min) / t_n;

function f(u, p, t)
  Δx = p[1];
  u₋₋ = circshift(u, 2);
  u₋ = circshift(u, 1);
  u₊ = circshift(u, -1);
  u₊₊ = circshift(u, -2);

  uₓ = (u₊ - u₋) ./ (2 * Δx);
  uₓₓₓ = (-u₋₋ .+ 2u₋ .- 2u₊ + u₊₊) ./ (2 * Δx^3);
  uₜ = -(6u .* uₓ) .- uₓₓₓ;
  return uₜ
end


function f3(u, p, t)
  u₋₋₋ = circshift(u, 3);
  u₋₋ = circshift(u, 2);
  u₋ = circshift(u, 1);
  u₊ = circshift(u, -1);
  u₊₊ = circshift(u, -2);
  u₊₊₊ = circshift(u, -3);

  uₓ = (1/12 * u₋₋ - 2/3 * u₋ + 2/3 * u₊ - 1/12 * u₊₊) ./ Δx;
  uₓₓₓ = (1/8 * u₋₋₋ - u₋₋ + 13/8 * u₋ - 13/8 * u₊ + u₊₊ - 1/12 * u₊₊₊) ./ (Δx^3);
  uₜ = -(6u .* uₓ) .- uₓₓₓ;
  return uₜ
end

function f2(u, p, t)
  k = p[1]
  @show(print(k))
  # u[1] = 0.
  # u[end] = 0.

  û = FFTW.fft(u)
  ûₓ = 1im .* k .* û
  ûₓₓₓ = -1im .* k.^3 .* û

  uₓ = FFTW.ifft(ûₓ)
  uₓₓₓ = FFTW.ifft(ûₓₓₓ)
  uₜ = (-6 * u .* uₓ) .- uₓₓₓ
  return real.(uₜ)
end

G(Δ, x) = √(6 / π) / Δ * exp(-6x^2 / Δ^2);
Δ = 4Δx;
W = sum(G.(Δ, x .- x' .- z) for z ∈ -2:2);
W = W ./ sum(W; dims = 2);

k = 2 * pi * AbstractFFTs.fftfreq(x_n, 1. / Δx) # Sampling rate, inverse of sample spacing

function high_dim_random_init2(t, x, m=28)
  d = Normal(0., 1.)
  nu = rand(d, 2 * m)
  x = (x .- x[1]) ./ (x[end] .- x[1])
  s = [nu[2 * k] * sin.(2 * pi * k * x) + nu[2 * k - 1] * cos.(2 * pi * k * x) for k in range(1, m, step=1)]

  u0 = zeros(Float64, size(t, 1), size(x, 1))
  u0[1, :] .= (1 / sqrt(m)) .* sum(s)
  # u0[:, 1] .= 0
  # u0[:, end] .= 0

  return u0
end

u0 = random_init(t, x)[1, :];
u0 = @. exp(-(x - 0.5)^2 / 0.005);
u0 = @. sinpi(2x) + sinpi(6x) + cospi(10x);
u0 = high_dim_random_init2(t, x, 3)[1, :];
Plots.plot(x, u0; label = "Unfiltered")


prob = ODEProblem(ODEFunction(f), copy(u0), extrema(t), (Δx));
sol = solve(prob, Rodas4P(), saveat=t, dt=0.01);
t2 = sol.t;
u = hcat(sol.u...);
Wsol = W * sol;
tmp = GraphicTools.show_state(u, t, x, "", "t", "x")
GraphicTools.show_state(Wsol, t, x, "", "t", "x")
GraphicTools.show_state(u .- Wsol, t, x, "", "t", "x")

# using PyPlot
# Plots.savefig(tmp, "test.png")

# pl = Plots.plot(; xlabel = "x", ylims = extrema(sol[:, :]))
# Plots.plot!(pl, x, Wsol[:, 2]; label = "Unfiltered")
# Plots.plot!(pl, x, Wsol[:, end]; label = "Unfiltered")

# for (i, t) ∈ enumerate(t)
#   pl = Plots.plot(; xlabel = "x", ylims = extrema(sol[:, :]))
#   Plots.plot!(pl, x, sol[i]; label = "Unfiltered")
#   Plots.plot!(pl, x, Wsol[:, i]; label = "Filtered")
#   display(pl)
#   # sleep(0.05) # Time for plot pane to update
# end


# ==== Generate + Fix data sets for KdV

include("../../utils/generators.jl");

snap_kwarg=(; t_max=10, t_min=0., x_max=8*pi, x_min=0., t_n=128, x_n=64, typ=2);
init_kwarg = (; mu=3);
# dataset2 = Generator.generate_kdv_dataset(1, 2, "", snap_kwarg, init_kwarg); # kdv_high_dim_m25_t10_128_x8pi_64_up2.jld2
dataset = Generator.read_dataset("dataset/kdv_high_dim_m25_t10_128_x8pi_64_up2.jld2")["training_set"];

fix = [];
for (i, data) in enumerate(dataset)
  if (size(data[2]) != (64, 128))
    push!(fix, i);
    println(i)
    println(size(data[2]))
  end
end

for (i, j) in enumerate(fix)
  println(j)
  println(i)
  println(size(dataset[j][2]));
  dataset[j] = dataset2[i];
  println(size(dataset[j][2]));
end

JLD2.save("dataset/kdv_high_dim_m25_t10_128_x8pi_64_up2.jld2", "training_set", dataset);

dataset = Generator.read_dataset("dataset/kdv_high_dim_m25_t0001_128_x1_64.jld2")["training_set"];
x = LinRange(x_min, 8*pi, 64);
t, u, _, _ = dataset[1];
display(GraphicTools.show_state(u, t, x, "", "t", "x"))

for (i, data) in enumerate(dataset)
  t, u, _, _ = data;
  display(GraphicTools.show_state(u, t, x, "", "t", "x"))
end

# =============