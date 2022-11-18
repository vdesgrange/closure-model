using FFTW
using AbstractFFTs
using OrdinaryDiffEq
using Plots
using LinearAlgebra
using SparseArrays
using Random
using Distributions
using JLD2

include("../../equations/initial_functions.jl")
include("../../utils/graphic_tools.jl")

function show_state(u, x, y, title, xlabel, ylabel)
  xₙ, yₙ = size(x)[1], size(y)[1];
  xₘᵢₙ, xₘₐₓ = x[1], x[end];
  yₘᵢₙ, yₘₐₓ = y[1], y[end];

  xformatter = x -> string(round(x / xₙ * xₘₐₓ + xₘᵢₙ, digits=2));
  yformatter = y -> string(round(y / yₙ * yₘₐₓ + yₘᵢₙ, digits=2));

  pl = heatmap(u);
  heatmap!(pl,
      title = title,
      # dpi=600,
      # aspect_ratio = :equal,
      # reuse=false,
      # c=:dense,
      xlabel=xlabel,
      ylabel=ylabel,
      # xticks=(1:7:size(x)[1], [xformatter(x) for x in 0:7:size(x)[1]]),
      # yticks=(1:7:size(y)[1], [yformatter(y) for y in 0:7:size(y)[1]]),
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


t_max = 1.; # 10
t_min = 0;
x_max = pi; # 8 * pi
x_min = 0;
x_n = 128;
t_n = 256;

# t = LinRange(0, 0.0002, 51)
t = LinRange(t_min, t_max, t_n);
x = LinRange(x_min, x_max, x_n);

Δx = (x_max - x_min) / x_n;
Δt = (t_max - t_min) / t_n;

function f(u, p, t)
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

u0 = random_init(t, x)[1, :];
u0 = @. exp(-(x - 0.5)^2 / 0.005);
u0 = @. sinpi(2x) + sinpi(6x) + cospi(10x);
u0 = InitialFunctions.high_dim_random_init2(t, x, 3)[1, :];
Plots.plot(x, u0; label = "Unfiltered")

prob = ODEProblem(ODEFunction(f), copy(u0), extrema(t), (k));
sol = solve(prob, Rodas4P(), saveat=t, dt=0.01);
t2 = sol.t;
u = hcat(sol.u...);
Wsol = W * sol;
tmp = GraphicTools.show_state(u, t, x, "", "t", "x")
GraphicTools.show_state(Wsol, t, x, "", "t", "x")
GraphicTools.show_state(u .- Wsol, t, x, "", "t", "x")

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
# kdv_high_dim_mu3_t10_128_x8pi_64_typ2_up1.jld2
snap_kwarg =(; t_max=5., t_min=0., x_max=4*pi, x_min=0., t_n=128, x_n=64, typ=2);
init_kwarg = (; mu=3);
# dataset = Generator.generate_kdv_dataset(128, 2, "", snap_kwarg, init_kwarg);
# dataset = Generator.read_dataset("dataset/kdv_high_dim_m25_t10_128_x8pi_64_up2.jld2")["training_set"];
t, u , _, _ = dataset[128];
x = LinRange(0, 4*pi, 64);
GraphicTools.show_state(u, t, x, "", "t", "x")

# fix = [];
# for (i, data) in enumerate(dataset2)
#   print(i)
#   if (size(data[2]) != (64, 128))
#     push!(fix, i);
#     println(i)
#     println(size(data[2]))
#   end
# end
# size(fix)
# for (i, j) in enumerate(fix)
#   println(j)
#   println(i)
#   println(size(dataset2[j][2]));
#   dataset2[j] = dataset3[i];
#   println(size(dataset2[j][2]));
# end

JLD2.save("kdv_high_dim_n128_m3_t5_128_x4pi_64_up2.jld2", "training_set", dataset);

# ============
