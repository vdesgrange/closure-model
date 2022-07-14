module GraphicTools

using Plots
pyplot()


function simple_plotter(ks, title="Simple plot", L=1.)
  plot()
  for k in ks
    x = LinRange(0, L, size(k, 1))
    plot!(x, k)
  end
end

function plot_regularization(noise, reg, a, b, c, d)
    plot(dpi=200)
    plot!(reg, a, markershape=:x, label="Interpolation")
    plot!(reg, b, markershape=:x, label="Full")
    plot!(reg, c, markershape=:x, label="Training")
    plot!(reg, d, markershape=:x, label="Validation")
    plot!(xlabel="Regularization", ylabel="Cost", xaxis=:log)
    plot!(xticks=(reg,["1e-12", "1e-10", "1e-8", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]))
end

function animate_plot(u, x)
end

function show_state(u, title, x="t", y="x")
    pl = heatmap(
        u,
        aspect_ratio = :equal,
        # xlims = (1 / 2, size(u, 2) + 1 / 2),
        # ylims = (1 / 2, size(u, 1) + 1 / 2),
    );
    heatmap!(pl,
        #xlabel = x,
        #ylabel = y,
        plot_title = title,
        dpi=200,
        reuse=false,
    );
    return pl
end

function show_err(a, b, title, x="x", y="t")
    pl = heatmap(
        abs.(a .- b),
        aspect_ratio = :equal,
        c = :dense,
    );
    heatmap!(pl,
        c = :devon,
        ylabel = y,
        xlabel = x,
        plot_title = title,
        dpi=200,
    );
    return pl
end

function domain_curve(k, v, L)
end

function visualize_u_from_F(F, t, u, u0)
end

end
