module Solver


"""
  third order TVD Runge-Kutta method
"""
function tvd_rk3(f, u, Δt)
  u₁ = u + Δt * f(u);
  u₂ = 0.75 * u + 0.25 * u₁ + 0.25 * Δt * f(u₁);
  u = (1.0/3.0) * u + (2.0/3.0) * u₂ + (2.0/3.0) * Δt  * f(u₂);
end

"""
  fourth order Runge-Kutta method
"""
function rk4(f, u, Δt)
  u₁ = Δt * f(u);
  u₂ = Δt * f(u + u₁ ./ 2);
  u₃ = Δt * f(u + u₂ ./ 2);
  u₄ = Δt * f(u + u₃);
  u = u + (1/6) * (u₁ + 2 .* u₂ + 2 .* u₃ +  u₄);
end

end
