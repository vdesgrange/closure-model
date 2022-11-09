# module BurgersCoarse

using FFTW
using AbstractFFTs
using Statistics
using Zygote
using Flux
using DiffEqFlux
using OrdinaryDiffEq
using BSON: @save

include("../../utils/processing_tools.jl")
include("../../utils/generators.jl")
include("../../utils/graphic_tools.jl")
include("../../equations/equations.jl")


snap_kwarg= repeat([(; tₘₐₓ=2., tₘᵢₙ=0., xₘₐₓ=pi, xₘᵢₙ=0., tₙ=64, xₙ=64, ν=0.04, typ=2)], 256);
init_kwarg = repeat([(; mu=5)], 256);
# dataset = Generator.generate_closure_dataset(256, 16, "./dataset/viscous_burgers_high_dim_t2_64_xpi_64_nu0.04_typ2_m10_256_up16_j173.jld2", snap_kwarg, init_kwarg);
x = LinRange(0, pi, 64);
for (i, data) in enumerate(dataset)
  t, u, _, _ = data;
  display(GraphicTools.show_state(u, t, x, "", "t", "x"))
end

dataset = Generator.read_dataset("./dataset/viscous_burgers_high_dim_t2_64_xpi_64_nu0.04_typ2_m10_256_up16_j173.jld2")["training_set"];

mse = 0;
rmse = 0;
for (i, data) in enumerate(dataset)
    t, u, args, _ = data;
    Δx = (args.xₘₐₓ - args.xₘᵢₙ) / (args.xₙ - 1);
    t₁, û = Equations.get_burgers_fft(t, Δx, args.xₙ, args.ν, u[:, 1]);
    display(GraphicTools.show_state(u, t, x, "", "t", "x"));
    # display(GraphicTools.show_state(û, t₁, x, "", "t", "x"))
    # display(GraphicTools.show_err(û, u, t₁, x, "", "t", "x"))
    tmp = Flux.mse(û, u)
    mse += tmp;
    rmse += sqrt(tmp);
    println(sqrt(tmp));
end

println(mse / 256);