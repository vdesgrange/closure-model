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
    );
    return pl
end

function show_err(a, title, x="x", y="t", lim=none, aspect=1)
end

function domain_curve(k, v, L)
end

function visualize_u_from_F(F, t, u, u0)
end

end
