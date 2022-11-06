include("../../utils/generators.jl");

snap_kwarg= repeat([(; tₘₐₓ=2., tₘᵢₙ=0., xₘₐₓ=pi, xₘᵢₙ=0., tₙ=64, xₙ=64, ν=0.04, typ=2)], 256);
init_kwarg = repeat([(; mu=10)], 256);
dataset = Generator.generate_closure_dataset(256, 16, "viscous_burgers_high_dim_m10_256_up16_j173.jld2", snap_kwarg, init_kwarg);
# dataset = Generator.read_dataset("kdv_high_dim_m25_t10_128_x30_64_up8.jld2")["training_set"];

dataset = Generator.read_dataset("viscous_burgers_high_dim_m10_256_up32_j173.jld2")["training_set"];
x = LinRange(x_min, x_max, 64);

for (i, data) in enumerate(dataset)
  t, u, _, _ = data;
  display(GraphicTools.show_state(u, t, x, "", "t", "x"))
end
