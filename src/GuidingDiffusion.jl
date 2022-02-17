module GuidingDiffusion


using Bridge
using StaticArrays
using Random 
using LinearAlgebra
using Distributions
using Bridge.Models
using DifferentialEquations
using Setfield
using ConstructionBase
using Interpolations
using IterTools
using Parameters

const sk=0
const ℝ{N} = SVector{N, Float64}
export ℝ
export @set!
export Vern7, Vern7direct, DE, properties, setproperties

include("tableaus_ode_solvers.jl")

include("types.jl")
export ParMove, Innovations, Observation, Htransform, Message, BackwardFilter, State

include("forwardguiding.jl")
export forwardguide

include("backwardfiltering.jl")
export backwardfiltering

include("utilities.jl")
export set_timegrids, say, printinfo

include("parameter_path_updates.jl")
export setpar, getpar, propose, logpriordiff, pcnupdate!, parupdate!, exploremoveσfixed!, parameterkernel, adjust_PNCparamters!

include("jansenrit.jl")

export JansenRitDiffusion, JansenRitDiffusionAux, wienertype

end # module
