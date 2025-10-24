using Checkpointing
using Enzyme

using CairoMakie, LinearAlgebra, Statistics, Random
using Parameters, UnPack, LaTeXStrings, NetCDF
using JLD2, DSP, FFTW
using ShallowWaters
using NLPModels, MadNLP

Random.seed!(649)

# Enzyme.API.looseTypeAnalysis!(true)
# Enzyme.API.strictAliasing!(true)

include("helper_functions.jl")
include("ensemble_kf.jl")