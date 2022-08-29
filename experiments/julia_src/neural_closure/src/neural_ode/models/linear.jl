using Flux

function LinearModel(x_n)
  """
    LinearModel(x_n)

  Create a linear flux chain model. Use for testing with linear heat equation.

  # Arguments
  - `x_n::Integer`: input/output dimension
  """
  return Flux.Chain(
    Flux.Dense(x_n, x_n, identity; bias=false, init=Flux.zeros32)
  );
end
