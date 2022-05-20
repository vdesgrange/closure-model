module GraphicTools

using PyCall
pushfirst!(pyimport("sys")."path", "")

utils = pyimport("graphic_tools.py")

end
