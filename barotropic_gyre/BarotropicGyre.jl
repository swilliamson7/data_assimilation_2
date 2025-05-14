using Checkpointing
using Enzyme

using CairoMakie, LinearAlgebra, Statistics, Random
using Parameters, UnPack, LaTeXStrings
using Optim, JLD2
using ShallowWaters

# Chosen random seed 649, will be used for all experiments 
Random.seed!(649)

# Enzyme.API.looseTypeAnalysis!(true)
# Enzyme.API.strictAliasing!(true)

include("helper_functions.jl")
include("ensemble_kf.jl")
include("exp1_initialcond_freqdata.jl")
include("exp2_sparseuandv_initcond.jl")
include("exp3_parameter_estimation.jl")