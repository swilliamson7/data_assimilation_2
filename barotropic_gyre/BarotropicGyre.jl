using Checkpointing
using Enzyme

using CairoMakie, LinearAlgebra, Statistics, Random
using Parameters, UnPack, LaTeXStrings
using Optim
using ShallowWaters

# Chosen random seed 649, will be used for all experiments 
Random.seed!(649)

include("helper_functions.jl")
include("ensemble_kf.jl")
include("exp1_initialcond_freqdata.jl")