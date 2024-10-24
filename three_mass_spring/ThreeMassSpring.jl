module ThreeMassSpring

using Plots, Enzyme, LinearAlgebra, Statistics, Random
using Parameters, UnPack, LaTeXStrings
using Optim

# Chosen random seed 649, will be used for all experiments 
Random.seed!(649)

include("structs.jl")
include("misc_functions.jl")
include("kalman_filter.jl")
include("adjoint.jl")
include("exp_1_fivedata.jl")
include("exp_1ish_consitentdata.jl")
include("exp_2_multdata.jl")
include("exp_3_obsofavg.jl")
include("exp_4_fixedpos.jl")
include("exp_5_param.jl")
include("exp_6_param_forcing.jl")
include("optim_functions.jl")
# include("optim_test_param.jl")

end