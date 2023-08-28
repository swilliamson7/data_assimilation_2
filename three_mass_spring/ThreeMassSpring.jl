module ThreeMassSpring

using Plots, Enzyme, LinearAlgebra, Statistics, Random
using Parameters, UnPack, LaTeXStrings
using Checkpointing, Zygote

include("misc_functions.jl")
include("structs.jl")
include("kalman_filter.jl")
include("adjoint.jl")
include("exp_1_fivedata.jl")
include("exp_1ish_consitentdata.jl")
include("exp_2_multdata.jl")
include("exp_3_obsofavg.jl")
include("exp_4_fixedpos.jl")

# Chosen random seed 649, will be used for all experiments 
Random.seed!(649)

end