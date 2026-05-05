mutable struct adjmodel{T, S} <: AbstractNLPModel{T,S}
    meta::NLPModelMeta{T,S}
    counters::Counters
    S::ShallowWaters.ModelSetup{T,T}        # model structure
    u_nohalo::Array{T,2}
    v_nohalo::Array{T,2}
    J::Float64                              # objective value
    data::Array{T, 2}
    data_steps::StepRange{Int, Int}         # when data is assimilated temporally
    data_spots::Array{Int,1}                # where data is located spatially, grid coordinates
    j::Int                                  # for keeping track of location in data
    i::Int                                  # timestep iterator, needed for checkpointing
    t::Int64                                # model time (e.g. dt * i)
end

@with_kw mutable struct enkfvars{T}

    nux::Int
    nuy::Int
    nvx::Int
    nvy::Int
    nx::Int
    ny::Int

    N::Int
    data_spots::Array{Int, 1}
    ldataspots::Int

    data_indexu::Array{Int,1} = data_spots
    data_indexv::Array{Int,1} = data_spots .+ nux*nuy
    data_indexeta::Array{Int,1} = data_spots .+ nux*nuy .+ nvx*nvy

    # variables that appear in construct of the ensemble, or when 
    # saving states
    uic::Array{T,2} = zeros(nux, nuy)
    vic::Array{T,2} = zeros(nvx, nvy)
    etaic::Array{T,2} = zeros(nx, ny)

    upert::Array{T,2} = zeros(nux, nuy)
    vpert::Array{T,2} = zeros(nvx, nvy)
    etapert::Array{T,2} = zeros(nx, ny)

    ekf_avgu::Array{T,2} = zeros(nux, nuy)
    ekf_avgv::Array{T,2} = zeros(nvx, nvy)
    ekf_avgeta::Array{T,2} = zeros(nx, ny)

    S_ensemble::Array{Any, 1} = zeros(N)

    # Kalman update variables

    Π::Array{T,2} = (I - (1 / N)*(ones(N) * ones(N)')) / sqrt(N - 1)
    U::Array{T,2} = zeros(ldataspots, N)

    Z::Array{T,2} = zeros(nux*nvx + nvx*nvy + nx*ny, N)

    E_cov::Array{T,2} = zeros(ldataspots, N)
    D::Array{T,2} = zeros(ldataspots, N)
    E::Array{T,2} = zeros(ldataspots, N)
    A::Array{T,2} = zeros(nux*nvx + nvx*nvy + nx*ny, N)
    Y::Array{T,2} = zeros(ldataspots, N)
    D̃::Array{T,2} = zeros(ldataspots, N)

end

function enkfvars{T}(G,Nensembles,data_spots,ldataspots) where {T<:AbstractFloat}

    @unpack nux,nuy,nvx,nvy,nx,ny = G

    return enkfvars{T}(nux=nux,nuy=nuy,nvx=nvx,nvy=nvy,nx=nx,ny=ny,
        N=Nensembles,data_spots=data_spots,ldataspots=ldataspots)
end

@with_kw mutable struct exp_initcond_ekfmodel{T}
    S::ShallowWaters.ModelSetup{T,T}        # model struct for ekf
    N::Int                                  # number of ensemble members
    data::Array{T, 2}                       # data to be assimilated
    sigma_initcond::T
    sigma_data::T
    sigma_forcing::T = 0.0                  # for the wind-stress experiment
    data_steps::StepRange{Int, Int}         # when data is assimilated
    data_spots::Array{Int, 1}               # where data is located, grid coordinates
    j::Int                                  # for keeping track of location in data
    t::Int64                                # model timestep (i * Δt)
    enkfvars::enkfvars{T}
end

# mutable struct exp_windstress_ekfmodel{T}
#     S::ShallowWaters.ModelSetup{T,T}        # model struct for adjoint
#     N::Int                                  # number of ensemble members
#     data::Array{T, 2}                       # data to be assimilated
#     sigma_initcond::T
#     sigma_data::T
#     sigma_forcing::T
#     pred_forcing::T
#     data_steps::StepRange{Int, Int}         # when data is assimilated
#     data_spots::Array{Int, 1}               # where data is located, grid coordinates
#     j::Int                                  # for keeping track of location in data
#     t::Int64                                # model timestep (i * Δt)
# end

