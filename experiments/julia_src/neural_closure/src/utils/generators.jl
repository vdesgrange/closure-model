module Generator

using FileIO
using JLD2
using Statistics
using Random


include("./equations/korteweg_de_vries.jl")
include("./equations/burgers.jl")
include("./equations/heat.jl")

function read_dataset(filepath)
  """
    read_dataset(filepath)
    Load JL2 data file.
  """
  training_set = JLD2.load(filepath)
  return training_set
end

end
