module BarotropicGyre

using Checkpointing#main
using Enzyme#main

using Plots, LinearAlgebra, Statistics, Random
using Parameters, UnPack, LaTeXStrings
using Optim

# Chosen random seed 649, will be used for all experiments 
Random.seed!(649)

include("./ShallowWaters.jl/src/ShallowWaters.jl")
include("ensemble_kf.jl")
include("experiment1_initialcond.jl")

end