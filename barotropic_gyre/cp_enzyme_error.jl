using Checkpointing
include("BarotropicGyre.jl")

mutable struct Chkp{T1,T2}
    S::ShallowWaters.ModelSetup{T1,T2}      # model structure
    data::Matrix{Float32}                   # computed data
    data_spots::Vector{Int}                 # location of data points spatially
    data_steps::StepRange{Int, Int}         # location of data points temporally
    J::Float64                              # objective function value
    j::Int                                  # for keeping track of location in data
    i::Int                                  # timestep iterator
    t::Float32                              # model time
end

function generate_data(S_true, data_spots, sigma_data; compute_freq=false)

    data = Float32.(zeros(length(data_spots), length(S_true.parameters.data_steps)))
    true_states = []

    # setup that happens prior to the actual integration
    Diag = S_true.Diag
    Prog = S_true.Prog

    @unpack u,v,η,sst = Prog
    @unpack u0,v0,η0 = Diag.RungeKutta
    @unpack u1,v1,η1 = Diag.RungeKutta
    @unpack du,dv,dη = Diag.Tendencies
    @unpack du_sum,dv_sum,dη_sum = Diag.Tendencies
    @unpack du_comp,dv_comp,dη_comp = Diag.Tendencies

    @unpack um,vm = Diag.SemiLagrange

    @unpack dynamics,RKo,RKs,tracer_advection = S_true.parameters
    @unpack time_scheme,compensated = S_true.parameters
    @unpack RKaΔt,RKbΔt = S_true.constants
    @unpack Δt_Δ,Δt_Δs = S_true.constants

    @unpack nt,dtint = S_true.grid
    @unpack nstep_advcor,nstep_diff,nadvstep,nadvstep_half = S_true.grid

    freqpoweru = []
    freqpowerv = []

    # calculate layer thicknesses for initial conditions
    ShallowWaters.thickness!(Diag.VolumeFluxes.h,η,S_true.forcing.H)
    ShallowWaters.Ix!(Diag.VolumeFluxes.h_u,Diag.VolumeFluxes.h)
    ShallowWaters.Iy!(Diag.VolumeFluxes.h_v,Diag.VolumeFluxes.h)
    ShallowWaters.Ixy!(Diag.Vorticity.h_q,Diag.VolumeFluxes.h)

    # calculate PV terms for initial conditions
    urhs = convert(Diag.PrognosticVarsRHS.u,u)
    vrhs = convert(Diag.PrognosticVarsRHS.v,v)
    ηrhs = convert(Diag.PrognosticVarsRHS.η,η)

    ShallowWaters.advection_coriolis!(urhs,vrhs,ηrhs,Diag,S_true)
    ShallowWaters.PVadvection!(Diag,S_true)

    # propagate initial conditions
    copyto!(u0,u)
    copyto!(v0,v)
    copyto!(η0,η)

    # store initial conditions of sst for relaxation
    copyto!(Diag.SemiLagrange.sst_ref,sst)
    j = 1

    for t = 1:S_true.grid.nt

        Diag = S_true.Diag
        Prog = S_true.Prog

        @unpack u,v,η,sst = Prog
        @unpack u0,v0,η0 = Diag.RungeKutta
        @unpack u1,v1,η1 = Diag.RungeKutta
        @unpack du,dv,dη = Diag.Tendencies
        @unpack du_sum,dv_sum,dη_sum = Diag.Tendencies
        @unpack du_comp,dv_comp,dη_comp = Diag.Tendencies

        @unpack um,vm = Diag.SemiLagrange

        @unpack dynamics,RKo,RKs,tracer_advection = S_true.parameters
        @unpack time_scheme,compensated = S_true.parameters
        @unpack RKaΔt,RKbΔt = S_true.constants
        @unpack Δt_Δ,Δt_Δs = S_true.constants

        @unpack nt,dtint = S_true.grid
        @unpack nstep_advcor,nstep_diff,nadvstep,nadvstep_half = S_true.grid
        i = S_true.parameters.i

        # ghost point copy for boundary conditions
        ShallowWaters.ghost_points!(u,v,η,S_true)
        copyto!(u1,u)
        copyto!(v1,v)
        copyto!(η1,η)

        if compensated
            fill!(du_sum,zero(Tprog))
            fill!(dv_sum,zero(Tprog))
            fill!(dη_sum,zero(Tprog))
        end

        for rki = 1:RKo
            if rki > 1
                ShallowWaters.ghost_points!(u1,v1,η1,S_true)
            end

            # type conversion for mixed precision
            u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
            v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
            η1rhs = convert(Diag.PrognosticVarsRHS.η,η1)

            ShallowWaters.rhs!(u1rhs,v1rhs,η1rhs,Diag,S_true,t)          # momentum only
            ShallowWaters.continuity!(u1rhs,v1rhs,η1rhs,Diag,S_true,t)   # continuity equation

            if rki < RKo
                ShallowWaters.caxb!(u1,u,RKbΔt[rki],du)   #u1 .= u .+ RKb[rki]*Δt*du
                ShallowWaters.caxb!(v1,v,RKbΔt[rki],dv)   #v1 .= v .+ RKb[rki]*Δt*dv
                ShallowWaters.caxb!(η1,η,RKbΔt[rki],dη)   #η1 .= η .+ RKb[rki]*Δt*dη
            end

            if compensated      # accumulate tendencies
                ShallowWaters.axb!(du_sum,RKaΔt[rki],du)
                ShallowWaters.axb!(dv_sum,RKaΔt[rki],dv)
                ShallowWaters.axb!(dη_sum,RKaΔt[rki],dη)
            else    # sum RK-substeps on the go
                ShallowWaters.axb!(u0,RKaΔt[rki],du)          #u0 .+= RKa[rki]*Δt*du
                ShallowWaters.axb!(v0,RKaΔt[rki],dv)          #v0 .+= RKa[rki]*Δt*dv
                ShallowWaters.axb!(η0,RKaΔt[rki],dη)          #η0 .+= RKa[rki]*Δt*dη
            end
        end

        if compensated
            # add compensation term to total tendency
            ShallowWaters.axb!(du_sum,-1,du_comp)
            ShallowWaters.axb!(dv_sum,-1,dv_comp)
            ShallowWaters.axb!(dη_sum,-1,dη_comp)

            ShallowWaters.axb!(u0,1,du_sum)   # update prognostic variable with total tendency
            ShallowWaters.axb!(v0,1,dv_sum)
            ShallowWaters.axb!(η0,1,dη_sum)

            ShallowWaters.dambmc!(du_comp,u0,u,du_sum)    # compute new compensation
            ShallowWaters.dambmc!(dv_comp,v0,v,dv_sum)
            ShallowWaters.dambmc!(dη_comp,η0,η,dη_sum)
        end

        ShallowWaters.ghost_points!(u0,v0,η0,S_true)

        # type conversion for mixed precision
        u0rhs = convert(Diag.PrognosticVarsRHS.u,u0)
        v0rhs = convert(Diag.PrognosticVarsRHS.v,v0)
        η0rhs = convert(Diag.PrognosticVarsRHS.η,η0)

        # ADVECTION and CORIOLIS TERMS
        # although included in the tendency of every RK substep,
        # only update every nstep_advcor steps if nstep_advcor > 0
        if dynamics == "nonlinear" && nstep_advcor > 0 && (i % nstep_advcor) == 0
            ShallowWaters.UVfluxes!(u0rhs,v0rhs,η0rhs,Diag,S_true)
            ShallowWaters.advection_coriolis!(u0rhs,v0rhs,η0rhs,Diag,S_true)
        end

        # DIFFUSIVE TERMS - SEMI-IMPLICIT EULER
        # use u0 = u^(n+1) to evaluate tendencies, add to u0 = u^n + rhs
        # evaluate only every nstep_diff time steps
        if (S_true.parameters.i % nstep_diff) == 0
            ShallowWaters.bottom_drag!(u0rhs,v0rhs,η0rhs,Diag,S_true)
            ShallowWaters.diffusion!(u0rhs,v0rhs,Diag,S_true)
            ShallowWaters.add_drag_diff_tendencies!(u0,v0,Diag,S_true)
            ShallowWaters.ghost_points_uv!(u0,v0,S_true)
        end

        # TRACER ADVECTION
        u0rhs = convert(Diag.PrognosticVarsRHS.u,u0)
        v0rhs = convert(Diag.PrognosticVarsRHS.v,v0)
        ShallowWaters.tracer!(i,u0rhs,v0rhs,Prog,Diag,S_true)

        # storing daily states for the "true" values
        if t ∈ 1:225:S_true.grid.nt
            temp1 = ShallowWaters.PrognosticVars{S_true.parameters.Tprog}(
                ShallowWaters.remove_halo(u,v,η,sst,S_true)...)
            push!(true_states, temp1)
        end

        # generating the data
        if t ∈ S_true.parameters.data_steps
            tempu = vec((ShallowWaters.PrognosticVars{S_true.parameters.Tprog}(
                ShallowWaters.remove_halo(u,v,η,sst,S_true)...)).u)
            tempv = vec((ShallowWaters.PrognosticVars{S_true.parameters.Tprog}(
                ShallowWaters.remove_halo(u,v,η,sst,S_true)...)).v)

            data[:, j] = Float32.([tempu; tempv][Int.(data_spots)] .+ sigma_data .* randn(length(data_spots)))
            j += 1
        end

        if compute_freq

            if t ∈ 10:10:S_true.grid.nt

                tempu = vec((ShallowWaters.PrognosticVars{S_true.parameters.Tprog}(
                    ShallowWaters.remove_halo(u,v,η,sst,S_true)...)).u)
                tempv = vec((ShallowWaters.PrognosticVars{S_true.parameters.Tprog}(
                    ShallowWaters.remove_halo(u,v,η,sst,S_true)...)).v)

                push!(freqpoweru, periodogram(states_adj[end].u; radialavg=true))
                push!(freqpowerv, periodogram(states_adj[end].v; radialavg=true))

            end

        end

        # Copy back from substeps
        copyto!(u,u0)
        copyto!(v,v0)
        copyto!(η,η0)

        t += dtint

    end

    return data, true_states

end

function cpintegrate(chkp, scheme)::Float64

    # calculate layer thicknesses for initial conditions
    ShallowWaters.thickness!(chkp.S.Diag.VolumeFluxes.h, chkp.S.Prog.η, chkp.S.forcing.H)
    ShallowWaters.Ix!(chkp.S.Diag.VolumeFluxes.h_u, chkp.S.Diag.VolumeFluxes.h)
    ShallowWaters.Iy!(chkp.S.Diag.VolumeFluxes.h_v, chkp.S.Diag.VolumeFluxes.h)
    ShallowWaters.Ixy!(chkp.S.Diag.Vorticity.h_q, chkp.S.Diag.VolumeFluxes.h)

    # calculate PV terms for initial conditions
    urhs = convert(chkp.S.Diag.PrognosticVarsRHS.u, chkp.S.Prog.u)
    vrhs = convert(chkp.S.Diag.PrognosticVarsRHS.v, chkp.S.Prog.v)
    ηrhs = convert(chkp.S.Diag.PrognosticVarsRHS.η, chkp.S.Prog.η)

    ShallowWaters.advection_coriolis!(urhs, vrhs, ηrhs, chkp.S.Diag, chkp.S)
    ShallowWaters.PVadvection!(chkp.S.Diag, chkp.S)

    # propagate initial conditions
    copyto!(chkp.S.Diag.RungeKutta.u0, chkp.S.Prog.u)
    copyto!(chkp.S.Diag.RungeKutta.v0, chkp.S.Prog.v)
    copyto!(chkp.S.Diag.RungeKutta.η0, chkp.S.Prog.η)

    # store initial conditions of sst for relaxation
    copyto!(chkp.S.Diag.SemiLagrange.sst_ref, chkp.S.Prog.sst)

    # run integration loop with checkpointing
    chkp.j = 1
    # @checkpoint_struct scheme chkp for chkp.i = 1:chkp.S.grid.nt
    for chkp.i = 1:chkp.S.grid.nt

        t = chkp.S.t
        i = chkp.i

        # ghost point copy for boundary conditions
        ShallowWaters.ghost_points!(chkp.S.Prog.u, chkp.S.Prog.v, chkp.S.Prog.η, chkp.S)
        copyto!(chkp.S.Diag.RungeKutta.u1, chkp.S.Prog.u)
        copyto!(chkp.S.Diag.RungeKutta.v1, chkp.S.Prog.v)
        copyto!(chkp.S.Diag.RungeKutta.η1, chkp.S.Prog.η)

        if chkp.S.parameters.compensated
            fill!(chkp.S.Diag.Tendencies.du_sum, zero(chkp.S.parameters.Tprog))
            fill!(chkp.S.Diag.Tendencies.dv_sum, zero(chkp.S.parameters.Tprog))
            fill!(chkp.S.Diag.Tendencies.dη_sum, zero(chkp.S.parameters.Tprog))
        end

        for rki = 1:chkp.S.parameters.RKo
            if rki > 1
                ShallowWaters.ghost_points!(
                    chkp.S.Diag.RungeKutta.u1,
                    chkp.S.Diag.RungeKutta.v1,
                    chkp.S.Diag.RungeKutta.η1,
                    chkp.S
                )
            end

            # type conversion for mixed precision
            u1rhs = convert(chkp.S.Diag.PrognosticVarsRHS.u, chkp.S.Diag.RungeKutta.u1)
            v1rhs = convert(chkp.S.Diag.PrognosticVarsRHS.v, chkp.S.Diag.RungeKutta.v1)
            η1rhs = convert(chkp.S.Diag.PrognosticVarsRHS.η, chkp.S.Diag.RungeKutta.η1)

            ShallowWaters.rhs!(u1rhs, v1rhs, η1rhs, chkp.S.Diag, chkp.S, t)          # momentum only
            ShallowWaters.continuity!(u1rhs, v1rhs, η1rhs, chkp.S.Diag, chkp.S, t)   # continuity equation

            if rki < chkp.S.parameters.RKo
                ShallowWaters.caxb!(
                    chkp.S.Diag.RungeKutta.u1,
                    chkp.S.Prog.u,
                    chkp.S.constants.RKbΔt[rki],
                    chkp.S.Diag.Tendencies.du
                )
                ShallowWaters.caxb!(
                    chkp.S.Diag.RungeKutta.v1,
                    chkp.S.Prog.v,
                    chkp.S.constants.RKbΔt[rki],
                    chkp.S.Diag.Tendencies.dv
                )
                ShallowWaters.caxb!(
                    chkp.S.Diag.RungeKutta.η1,
                    chkp.S.Prog.η,
                    chkp.S.constants.RKbΔt[rki],
                    chkp.S.Diag.Tendencies.dη
                )
            end

            if chkp.S.parameters.compensated
                ShallowWaters.axb!(chkp.S.Diag.Tendencies.du_sum, chkp.S.constants.RKaΔt[rki], chkp.S.Diag.Tendencies.du)
                ShallowWaters.axb!(chkp.S.Diag.Tendencies.dv_sum, chkp.S.constants.RKaΔt[rki], chkp.S.Diag.Tendencies.dv)
                ShallowWaters.axb!(chkp.S.Diag.Tendencies.dη_sum, chkp.S.constants.RKaΔt[rki], chkp.S.Diag.Tendencies.dη)
            else
                ShallowWaters.axb!(
                    chkp.S.Diag.RungeKutta.u0,
                    chkp.S.constants.RKaΔt[rki],
                    chkp.S.Diag.Tendencies.du
                )
                ShallowWaters.axb!(
                    chkp.S.Diag.RungeKutta.v0,
                    chkp.S.constants.RKaΔt[rki],
                    chkp.S.Diag.Tendencies.dv
                )
                ShallowWaters.axb!(
                    chkp.S.Diag.RungeKutta.η0,
                    chkp.S.constants.RKaΔt[rki],
                    chkp.S.Diag.Tendencies.dη
                )
            end
        end

        if chkp.S.parameters.compensated
            ShallowWaters.axb!(chkp.S.Diag.Tendencies.du_sum, -1, chkp.S.Diag.Tendencies.du_comp)
            ShallowWaters.axb!(chkp.S.Diag.Tendencies.dv_sum, -1, chkp.S.Diag.Tendencies.dv_comp)
            ShallowWaters.axb!(chkp.S.Diag.Tendencies.dη_sum, -1, chkp.S.Diag.Tendencies.dη_comp)

            ShallowWaters.axb!(chkp.S.Diag.RungeKutta.u0, 1, chkp.S.Diag.Tendencies.du_sum)
            ShallowWaters.axb!(chkp.S.Diag.RungeKutta.v0, 1, chkp.S.Diag.Tendencies.dv_sum)
            ShallowWaters.axb!(chkp.S.Diag.RungeKutta.η0, 1, chkp.S.Diag.Tendencies.dη_sum)

            ShallowWaters.dambmc!(
                chkp.S.Diag.Tendencies.du_comp,
                chkp.S.Diag.RungeKutta.u0,
                chkp.S.Prog.u,
                chkp.S.Diag.Tendencies.du_sum
            )
            ShallowWaters.dambmc!(
                chkp.S.Diag.Tendencies.dv_comp,
                chkp.S.Diag.RungeKutta.v0,
                chkp.S.Prog.v,
                chkp.S.Diag.Tendencies.dv_sum
            )
            ShallowWaters.dambmc!(
                chkp.S.Diag.Tendencies.dη_comp,
                chkp.S.Diag.RungeKutta.η0,
                chkp.S.Prog.η,
                chkp.S.Diag.Tendencies.dη_sum
            )
        end

        ShallowWaters.ghost_points!(
            chkp.S.Diag.RungeKutta.u0,
            chkp.S.Diag.RungeKutta.v0,
            chkp.S.Diag.RungeKutta.η0,
            chkp.S
        )

        u0rhs = convert(chkp.S.Diag.PrognosticVarsRHS.u, chkp.S.Diag.RungeKutta.u0)
        v0rhs = convert(chkp.S.Diag.PrognosticVarsRHS.v, chkp.S.Diag.RungeKutta.v0)
        η0rhs = convert(chkp.S.Diag.PrognosticVarsRHS.η, chkp.S.Diag.RungeKutta.η0)

        if chkp.S.parameters.dynamics == "nonlinear" && chkp.S.grid.nstep_advcor > 0 && (i % chkp.S.grid.nstep_advcor) == 0
            ShallowWaters.UVfluxes!(u0rhs, v0rhs, η0rhs, chkp.S.Diag, chkp.S)
            ShallowWaters.advection_coriolis!(u0rhs, v0rhs, η0rhs, chkp.S.Diag, chkp.S)
        end

        if (chkp.S.parameters.i % chkp.S.grid.nstep_diff) == 0
        ShallowWaters.bottom_drag!(u0rhs, v0rhs, η0rhs, chkp.S.Diag, chkp.S)
        ShallowWaters.diffusion!(u0rhs, v0rhs, chkp.S.Diag, chkp.S)
        ShallowWaters.add_drag_diff_tendencies!(
            chkp.S.Diag.RungeKutta.u0,
            chkp.S.Diag.RungeKutta.v0,
            chkp.S.Diag,
            chkp.S
        )
        ShallowWaters.ghost_points_uv!(
            chkp.S.Diag.RungeKutta.u0,
            chkp.S.Diag.RungeKutta.v0,
            chkp.S
        )
    end

    t += chkp.S.grid.dtint

    u0rhs = convert(chkp.S.Diag.PrognosticVarsRHS.u, chkp.S.Diag.RungeKutta.u0)
    v0rhs = convert(chkp.S.Diag.PrognosticVarsRHS.v, chkp.S.Diag.RungeKutta.v0)
    ShallowWaters.tracer!(i, u0rhs, v0rhs, chkp.S.Prog, chkp.S.Diag, chkp.S)

    if i in chkp.data_steps
        temp = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(
            chkp.S.Prog.u,
            chkp.S.Prog.v,
            chkp.S.Prog.η,
            chkp.S.Prog.sst,
            chkp.S
        )...)

        tempuv = [vec(temp.u); vec(temp.v)][chkp.data_spots]

        chkp.J += sum((tempuv - chkp.data[:, chkp.j]).^2)

        chkp.j += 1
    end

    copyto!(chkp.S.Prog.u, chkp.S.Diag.RungeKutta.u0)
    copyto!(chkp.S.Prog.v, chkp.S.Diag.RungeKutta.v0)
    copyto!(chkp.S.Prog.η, chkp.S.Diag.RungeKutta.η0)
    end

    return chkp.J

end

function gradient_eval(G, param_guess, data, data_spots, data_steps, Ndays)

    # Type precision
    T = Float32

    P = ShallowWaters.Parameter(T=T;
        output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        tracer_advection=false,
        tracer_relaxation=false,
        bottom_drag="quadratic",
        seasonal_wind_x=false,
        data_steps=data_steps,
        topography="flat",
        bc="nonperiodic",
        α=2,
        nx=128,
        Ndays=Ndays,
        initial_cond="ncfile",
        initpath="./data_files_forkf/128_spinup_noforcing/"
    )

    S = ShallowWaters.model_setup(P)

    S.Prog.u = reshape(param_guess[1:17292], 131, 132)
    S.Prog.v = reshape(param_guess[17293:34584], 132, 131)
    S.Prog.η = reshape(param_guess[34585:end-1], 130, 130)
    S.parameters.Fx0 = param_guess[end]

    snaps = Int(floor(sqrt(S.grid.nt)))
    revolve = Revolve{exp3_Chkp{T, T}}(S.grid.nt,
        snaps;
        verbose=1,
        gc=true,
        write_checkpoints=false,
        write_checkpoints_filename = "",
        write_checkpoints_period = 224
    )
    data_spots = Int.(data_spots)

    chkp = exp3_Chkp{T, T}(S,
        data,
        data_spots,
        data_steps,
        0.0,
        1,
        1,
        0.0
    )
    dchkp = Enzyme.make_zero(chkp)

    chkp_prim = deepcopy(chkp)
    J = exp3_cpintegrate(chkp_prim, revolve)
    println("Cost without AD: $J")

    @time J = autodiff(
        set_runtime_activity(Enzyme.ReverseWithPrimal),
        exp3_cpintegrate,
        Active,
        Duplicated(chkp, dchkp),
        Const(revolve)
    )[2]
    println("Cost with AD: $J")

    # Get gradient
    @unpack u, v, η = dchkp.S.Prog
    G .= [vec(u); vec(v); vec(η); dchkp.S.parameters.Fx0]

    return chkp, dchkp, G

end

function run()

    xu = 30:10:100
    yu = 40:10:100
    Xu = xu' .* ones(length(yu))
    Yu = ones(length(xu))' .* yu

    sigma_data = 0.01
    sigma_initcond = 0.02
    data_steps = 220:220:6733
    data_spotsu = vec((Xu.-1) .* 127 + Yu)
    data_spotsv = vec((Xu.-1) .* 128 + Yu) .+ (128*127)        # just adding the offset of the size of u, otherwise same spatial locations roughly
    data_spots = [data_spotsu; data_spotsv]
    Ndays = 1

    P_pred = ShallowWaters.Parameter(T=Float32;
    output=false,
    L_ratio=1,
    g=9.81,
    H=500,
    wind_forcing_x="double_gyre",
    Lx=3840e3,
    tracer_advection=false,
    tracer_relaxation=false,
    bottom_drag="quadratic",
    seasonal_wind_x=false,
    data_steps=data_steps,
    topography="flat",
    bc="nonperiodic",
    α=2,
    nx=128,
    Ndays=Ndays,
    initial_cond="ncfile",
    initpath="./data_files_forkf/128_spinup_noforcing/")

    S_true = ShallowWaters.model_setup(P_pred)

    data, _ = generate_data(S_true, data_spots, sigma_data)

    S_pred = ShallowWaters.model_setup(P_pred)

    S_pred.parameters.Fx0 = .0001 * S_true.parameters.Fx0
    Prog_pred = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(S_pred.Prog.u,
        S_pred.Prog.v,
        S_pred.Prog.η,
        S_pred.Prog.sst,
        S_pred)...
    )

    # perturb initial conditions from those seen by the "true" model (create incorrect initial conditions)
    Prog_pred.u = Prog_pred.u + sigma_initcond .* randn(size(Prog_pred.u))
    Prog_pred.v = Prog_pred.v + sigma_initcond .* randn(size(Prog_pred.v))
    Prog_pred.η = Prog_pred.η + sigma_initcond .* randn(size(Prog_pred.η))

    uic,vic,etaic = ShallowWaters.add_halo(Prog_pred.u,Prog_pred.v,Prog_pred.η,Prog_pred.sst,S_pred)
    S_pred.Prog.u = uic
    S_pred.Prog.v = vic
    S_pred.Prog.η = etaic

    param_guess = [vec(uic); vec(vic); vec(etaic); S_pred.parameters.Fx0]

    Ndays = copy(P_pred.Ndays)
    data_steps = copy(P_pred.data_steps)

    dS = Enzyme.Compiler.make_zero(S_pred)
    G = zeros(length(dS.Prog.u) + length(dS.Prog.v) + length(dS.Prog.η) + 1)

    chkp, dchkp, G = gradient_eval(G, param_guess, data, data_spots, data_steps, Ndays)

    # fg!_closure(F, G, ic) = exp2_FG(F, G, ic, data, data_spots, data_steps, Ndays)
    # obj_fg = Optim.only_fg!(fg!_closure)
    # result = Optim.optimize(obj_fg, param_guess, Optim.LBFGS(), Optim.Options(show_trace=true, iterations=1))

    return chkp, dchkp, G

end

chkp, dchkp, G = run()
@show norm(G[end])

function finitedifference(x_coord, y_coord)

    Ndays = 1

    xu = 30:10:100
    yu = 40:10:100
    Xu = xu' .* ones(length(yu))
    Yu = ones(length(xu))' .* yu

    sigma_data = 0.01
    data_steps = 220:220:6733
    data_spotsu = vec((Xu.-1) .* 127 + Yu)
    data_spotsv = vec((Xu.-1) .* 128 + Yu) .+ (128*127)
    data_spots = [data_spotsu; data_spotsv]

    P1 = ShallowWaters.Parameter(T=Float32;
    output=false,
    L_ratio=1,
    g=9.81,
    H=500,
    wind_forcing_x="double_gyre",
    Lx=3840e3,
    tracer_advection=false,
    tracer_relaxation=false,
    bottom_drag="quadratic",
    seasonal_wind_x=false,
    data_steps=data_steps,
    topography="flat",
    bc="nonperiodic",
    α=2,
    nx=128,
    Ndays=Ndays,
    initial_cond="ncfile",
    initpath="./data_files_forkf/128_spinup_noforcing/")

    S_true = ShallowWaters.model_setup(P1)
    data, _ = generate_data(S_true, data_spots, sigma_data)

    snaps = Int(floor(sqrt(S_true.grid.nt)))
    revolve = Revolve{ShallowWaters.ModelSetup}(S_true.grid.nt,
        snaps;
        verbose=1,
        gc=true,
        write_checkpoints=false,
        write_checkpoints_filename = "",
        write_checkpoints_period = 224
    )

    chkp = Chkp{T, T}(S_true,
        data,
        data_spots,
        data_steps,
        0.0,
        1,
        1,
        0.0
    )
    dchkp = Enzyme.make_zero(chkp)

    chkp_prim = deepcopy(chkp)
    J = cpintegrate(chkp_prim, revolve)
    dS_outer = Enzyme.Compiler.make_zero(S_outer)

    ddata = Enzyme.make_zero(data)
    ddata_spots = Enzyme.make_zero(data_spots)

    autodiff(set_runtime_activity(Enzyme.ReverseWithPrimal), cpintegrate,
        Duplicated(S_outer, dS_outer),
        Const(revolve),
        Duplicated(data,ddata),
        Duplicated(data_spots, ddata_spots)
    )

    enzyme_deriv = dS_outer.Prog.u[x_coord, y_coord]

    steps = [50, 40, 30, 20, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

    P2 = ShallowWaters.Parameter(T=Float32;output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        tracer_advection=false,
        tracer_relaxation=false,
        bottom_drag="quadratic",
        seasonal_wind_x=false,
        data_steps=data_steps,
        topography="flat",
        bc="nonperiodic",
        α=2,
        nx=128,
        Ndays=Ndays,
        initial_cond="ncfile",
        initpath="./data_files_forkf/128_spinup_noforcing/")

    S = ShallowWaters.model_setup(P2)
    J_outer = cpintegrate(S, revolve)

    diffs = []

    for s in steps

        P = ShallowWaters.Parameter(T=Float32;output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        tracer_advection=false,
        tracer_relaxation=false,
        bottom_drag="quadratic",
        seasonal_wind_x=false,
        data_steps=data_steps,
        topography="flat",
        bc="nonperiodic",
        α=2,
        nx=128,
        Ndays=Ndays,
        initial_cond="ncfile",
        initpath="./data_files_forkf/128_spinup_noforcing/")

        S_inner = ShallowWaters.model_setup(P)

        S_inner.Prog.u[x_coord, y_coord] += s

        J_inner = cpintegrate(S_inner, revolve)

        push!(diffs, (J_inner - J_outer) / s)

    end

    return diffs, enzyme_deriv

end

# diffs, enzyme_deriv = finitedifference(20, 30)
