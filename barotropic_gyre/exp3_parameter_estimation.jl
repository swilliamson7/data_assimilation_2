mutable struct exp3_Chkp{T1,T2}
    S::ShallowWaters.ModelSetup{T1,T2}      # model structure
    data::Matrix{Float32}                   # computed data
    data_spots::Vector{Int}                 # location of data points spatially
    data_steps::StepRange{Int, Int}         # location of data points temporally
    J::Float64                              # objective function value
    j::Int                                  # for keeping track of location in data
    i::Int                                  # timestep iterator
    t::Int64                                # model time
end

function exp3_generate_data(S_true, data_spots, sigma_data)

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
        if t ∈ 10:10:S_true.grid.nt
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

        # Copy back from substeps
        copyto!(u,u0)
        copyto!(v,v0)
        copyto!(η,η0)

        t += dtint

    end

    return data, true_states

end

function exp3_cpintegrate(chkp, scheme)::Float64

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
    @checkpoint_struct scheme chkp for chkp.i = 1:chkp.S.grid.nt

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

function exp3_integrate(S, data, data_spots)

    # setup
    Diag = S.Diag
    Prog = S.Prog

    @unpack u,v,η,sst = Prog
    @unpack u0,v0,η0 = Diag.RungeKutta
    @unpack u1,v1,η1 = Diag.RungeKutta
    @unpack du,dv,dη = Diag.Tendencies
    @unpack du_sum,dv_sum,dη_sum = Diag.Tendencies
    @unpack du_comp,dv_comp,dη_comp = Diag.Tendencies

    @unpack um,vm = Diag.SemiLagrange

    @unpack dynamics,RKo,RKs,tracer_advection = S.parameters
    @unpack time_scheme,compensated = S.parameters
    @unpack RKaΔt,RKbΔt = S.constants
    @unpack Δt_Δ,Δt_Δs = S.constants

    @unpack nt,dtint = S.grid
    @unpack nstep_advcor,nstep_diff,nadvstep,nadvstep_half = S.grid

    # calculate layer thicknesses for initial conditions
    ShallowWaters.thickness!(Diag.VolumeFluxes.h,η,S.forcing.H)
    ShallowWaters.Ix!(Diag.VolumeFluxes.h_u,Diag.VolumeFluxes.h)
    ShallowWaters.Iy!(Diag.VolumeFluxes.h_v,Diag.VolumeFluxes.h)
    ShallowWaters.Ixy!(Diag.Vorticity.h_q,Diag.VolumeFluxes.h)

    # calculate PV terms for initial conditions
    urhs = convert(Diag.PrognosticVarsRHS.u,u)
    vrhs = convert(Diag.PrognosticVarsRHS.v,v)
    ηrhs = convert(Diag.PrognosticVarsRHS.η,η)

    ShallowWaters.advection_coriolis!(urhs,vrhs,ηrhs,Diag,S)
    ShallowWaters.PVadvection!(Diag,S)

    # propagate initial conditions
    copyto!(u0,u)
    copyto!(v0,v)
    copyto!(η0,η)

    # store initial conditions of sst for relaxation
    copyto!(Diag.SemiLagrange.sst_ref,sst)
    j = 1

    # run integration loop with checkpointing
    for S.parameters.i = 1:S.grid.nt

        Diag = S.Diag
        Prog = S.Prog
    
        @unpack u,v,η,sst = Prog
        @unpack u0,v0,η0 = Diag.RungeKutta
        @unpack u1,v1,η1 = Diag.RungeKutta
        @unpack du,dv,dη = Diag.Tendencies
        @unpack du_sum,dv_sum,dη_sum = Diag.Tendencies
        @unpack du_comp,dv_comp,dη_comp = Diag.Tendencies
    
        @unpack um,vm = Diag.SemiLagrange
    
        @unpack dynamics,RKo,RKs,tracer_advection = S.parameters
        @unpack time_scheme,compensated = S.parameters
        @unpack RKaΔt,RKbΔt = S.constants
        @unpack Δt_Δ,Δt_Δs = S.constants
    
        @unpack nt,dtint = S.grid
        @unpack nstep_advcor,nstep_diff,nadvstep,nadvstep_half = S.grid
        t = S.t
        i = S.parameters.i

        # ghost point copy for boundary conditions
        ShallowWaters.ghost_points!(u,v,η,S)
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
                ShallowWaters.ghost_points!(u1,v1,η1,S)
            end

            # type conversion for mixed precision
            u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
            v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
            η1rhs = convert(Diag.PrognosticVarsRHS.η,η1)

            ShallowWaters.rhs!(u1rhs,v1rhs,η1rhs,Diag,S,t)          # momentum only
            ShallowWaters.continuity!(u1rhs,v1rhs,η1rhs,Diag,S,t)   # continuity equation

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

        ShallowWaters.ghost_points!(u0,v0,η0,S)

        # type conversion for mixed precision
        u0rhs = convert(Diag.PrognosticVarsRHS.u,u0)
        v0rhs = convert(Diag.PrognosticVarsRHS.v,v0)
        η0rhs = convert(Diag.PrognosticVarsRHS.η,η0)

        # ADVECTION and CORIOLIS TERMS
        # although included in the tendency of every RK substep,
        # only update every nstep_advcor steps if nstep_advcor > 0
        if dynamics == "nonlinear" && nstep_advcor > 0 && (i % nstep_advcor) == 0
            ShallowWaters.UVfluxes!(u0rhs,v0rhs,η0rhs,Diag,S)
            ShallowWaters.advection_coriolis!(u0rhs,v0rhs,η0rhs,Diag,S)
        end

        # DIFFUSIVE TERMS - SEMI-IMPLICIT EULER
        # use u0 = u^(n+1) to evaluate tendencies, add to u0 = u^n + rhs
        # evaluate only every nstep_diff time steps
        if (S.parameters.i % nstep_diff) == 0
            ShallowWaters.bottom_drag!(u0rhs,v0rhs,η0rhs,Diag,S)
            ShallowWaters.diffusion!(u0rhs,v0rhs,Diag,S)
            ShallowWaters.add_drag_diff_tendencies!(u0,v0,Diag,S)
            ShallowWaters.ghost_points_uv!(u0,v0,S)
        end

        t += dtint

        # TRACER ADVECTION
        u0rhs = convert(Diag.PrognosticVarsRHS.u,u0) 
        v0rhs = convert(Diag.PrognosticVarsRHS.v,v0)
        ShallowWaters.tracer!(i,u0rhs,v0rhs,Prog,Diag,S)

        # Cost function evaluation

        if S.parameters.i in S.parameters.data_steps

            temp = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(S.Prog.u,
            S.Prog.v,
            S.Prog.η,
            S.Prog.sst,S)...)

            tempuv = [vec(temp.u);vec(temp.v)][Int.(data_spots)]

            S.parameters.J += sum((tempuv - data[:, j]).^2)

            j += 1

        end

        # Copy back from substeps
        copyto!(u,u0)
        copyto!(v,v0)
        copyto!(η,η0)

    end

    return S.parameters.J

end

function exp3_cost_eval(param_guess, data, data_spots, data_steps, Ndays)

    P = ShallowWaters.Parameter(T=Float32;
    output=false,
    L_ratio=1,
    g=9.81,
    H=500,
    wind_forcing_x="double_gyre",
    Lx=3840e3,
    tracer_advection=false,
    tracer_relaxation=false,
    seasonal_wind_x=false,
    data_steps=data_steps,
    topography="flat",
    bc="nonperiodic",
    bottom_drag="quadratic",
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

    J = exp3_integrate(S, data, data_spots)

    return J

end

function exp3_gradient_eval(G, param_guess, data, data_spots, data_steps, Ndays)

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

    return nothing

end

function exp3_FG(F, G, param_guess, data, data_spots, data_steps, Ndays)

    G === nothing || exp3_gradient_eval(G, param_guess, data, data_spots, data_steps, Ndays)
    F === nothing || return exp3_cost_eval(param_guess, data, data_spots, data_steps, Ndays)

end

function exp3_bottomdrag_initialcond(N, data_spots, sigma_initcond, sigma_data; kwargs...)

    P_true = ShallowWaters.Parameter(T=Float32;kwargs...)

    S_true = ShallowWaters.model_setup(P_true)

    data, true_states = exp3_generate_data(S_true, data_spots, sigma_data)

    # initially setting up the prediction model with correct parameters
    S_pred = ShallowWaters.model_setup(P_true)

    for t = 1:224*3
        _ = one_step_function(S_pred)
    end

    # here we perturb the ones that are going to be the "incorrect" (perturbed) parameters

    S_pred.parameters.Fx0 = .0001 * S_true.parameters.Fx0
    Prog_pred = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(S_pred.Prog.u,
        S_pred.Prog.v,
        S_pred.Prog.η,
        S_pred.Prog.sst,
        S_pred)...
    )

    # perturb initial conditions from those seen by the "true" model (create incorrect initial conditions)
    upert = zeros(size(Prog_pred.u))
    vpert = zeros(size(Prog_pred.v))
    etapert = zeros(size(Prog_pred.η))

    for k = 1:127
    for j = 1:128

        for n = 1:20
            for m = 1:20
                upert[k,j] = sigma_initcond * randn(1)[1] * cos((pi * n / 127) * k + (pi * m / 128) * j)
                    + sigma_initcond * randn(1)[1] * sin((pi * n / 127) * k + (pi * m / 128) * j)
                vpert[j,k] = sigma_initcond * randn(1)[1] * cos((pi * n / 128) * j + (pi * m / 127) * k)
                    + sigma_initcond * randn(1)[1] * sin((pi * n / 128) * j + (pi * m / 127) * k)
            end
        end

    end
    end

    for k = 1:128
    for j = 1:128

        for n = 1:20
            for m = 1:20
                etapert[k,j] = sigma_initcond * randn(1)[1] * cos((pi * n / 128) * k + (pi * m / 128) * j)
                    + sigma_initcond * randn(1)[1] * sin((pi * n / 128) * k + (pi * m / 128) * j)
            end
        end

    end
    end

    Prog_pred.u = Prog_pred.u + upert
    Prog_pred.v = Prog_pred.v + vpert
    Prog_pred.η = Prog_pred.η + etapert
    
    uic,vic,etaic = ShallowWaters.add_halo(Prog_pred.u,Prog_pred.v,Prog_pred.η,Prog_pred.sst,S_pred)
    S_pred.Prog.u = uic
    S_pred.Prog.v = vic
    S_pred.Prog.η = etaic

    param_guess = [vec(uic); vec(vic); vec(etaic); S_pred.parameters.Fx0]

    S_kf_all, ekf_avgu, ekf_avgv = exp3_run_ensemble_kf(N,
    data,
    param_guess,
    data_spots,
    sigma_initcond,
    sigma_data;
    kwargs...
    )

    dS = Enzyme.Compiler.make_zero(S_pred)
    G = zeros(length(dS.Prog.u) + length(dS.Prog.v) + length(dS.Prog.η) + 1)

    Ndays = copy(S_pred.parameters.Ndays)
    data_steps = copy(S_pred.parameters.data_steps)

    fg!_closure(F, G, ic) = exp3_FG(F, G, ic, data, data_spots, data_steps, Ndays)
    obj_fg = Optim.only_fg!(fg!_closure)
    result = Optim.optimize(obj_fg, param_guess, Optim.LBFGS(), Optim.Options(show_trace=true, iterations=3))

    # jldsave("exp3_minimizer_initcond_forcing_adjoint_042925.jld2",
    #     u = reshape(result.minimizer[1:17292], 131, 132),
    #     v = reshape(result.minimizer[17293:34584], 132, 131),
    #     eta = reshape(result.minimizer[34585:end-1], 130, 130),
    #     Fx0 = result.minimizer[end]
    # )

    S_adj = ShallowWaters.model_setup(P_true)
    S_adj.Prog.u = reshape(result.minimizer[1:17292], 131, 132)
    S_adj.Prog.v = reshape(result.minimizer[17293:34584], 132, 131)
    S_adj.Prog.η = reshape(result.minimizer[34585:end-1], 130, 130)
    S_adj.parameters.Fx0 = result.minimizer[end]
    _, states_adj = exp3_generate_data(S_adj, data_spots, sigma_data)

    return S_kf_all, result, ekf_avgu, ekf_avgv, data, true_states, S_adj, states_adj

end

function run_exp3()

    xu = 30:10:100
    yu = 40:10:100
    Xu = xu' .* ones(length(yu))
    Yu = ones(length(xu))' .* yu

    N = 10
    sigma_data = 0.01
    sigma_initcond = 0.02
    data_steps = 220:220:6733
    data_spotsu = vec((Xu.-1) .* 127 + Yu)
    data_spotsv = vec((Xu.-1) .* 128 + Yu) .+ (128*127)        # just adding the offset of the size of u, otherwise same spatial locations roughly
    data_spots = [data_spotsu; data_spotsv]
    Ndays = 30

    S_kf_all, result, ekf_avgu, ekf_avgv, data, true_states, S_adj, states_adj = exp3_bottomdrag_initialcond(N,
        data_spots,
        sigma_initcond,
        sigma_data,
        output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        bottom_drag="quadratic",
        Lx=3840e3,
        tracer_advection=false,
        tracer_relaxation=false,
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

    return S_kf_all, result, ekf_avgu, ekf_avgv, data, true_states, S_adj, states_adj

end

function exp3_plots()

    # S_kf_all, Progkf_all, G, dS, data, states_true, result, S_adj, states_adj

    xu = 30:10:100
    yu = 40:10:100
    Xu = xu' .* ones(length(yu))
    Yu = ones(length(xu))' .* yu

    N = 10
    sigma_data = 0.01
    sigma_initcond = 0.02
    data_steps = 220:220:6733
    data_spotsu = vec((Xu.-1) .* 127 + Yu)
    data_spotsv = vec((Xu.-1) .* 128 + Yu) .+ (128*127)        # just adding the offset of the size of u, otherwise same spatial locations roughly
    data_spots = [data_spotsu; data_spotsv]
    Ndays = 30

    fig1 = Figure(size=(1000, 500));
    ax1, hm1 = heatmap(fig1[1,1], states_true[end].u,
        colormap=:balance,
        colorrange=(-maximum(states_true[end].u),
        maximum(states_true[end].u)),
        axis=(xlabel=L"x", ylabel=L"y", title=L"u(t = 30 \; \text{days}, x, y)"),
    );
    scatter!(ax1, vec(X), vec(Y), color=:green);
    Colorbar(fig1[1,2], hm1)

    
    kf_avgu = zeros(size(states_adj[1].u))
    kf_avgv = zeros(size(states_adj[1].v))
    for n = 1:10
        kf_avgu = kf_avgu .+ Progkf_all[end][n].u
        kf_avgv = kf_avgv .+ Progkf_all[end][n].v
    end
    kf_avgu = kf_avgu ./ 10
    kf_avgv = kf_avgv ./ 10

    fig1 = Figure(size=(800,700));
    ax1, hm1 = heatmap(fig1[1,1], states_adj[end].v,
    colormap=:balance,
    colorrange=(-maximum(states_adj[end].v),
    maximum(states_adj[end].v)),
    axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{v}(t = 30 \; \text{days}, x, y, +)")
    );
    Colorbar(fig1[1,2], hm1)
    ax2, hm2 = heatmap(fig1[1,3], abs.(states_true[end].v .- states_adj[end].v),
    colormap=:amp,
    colorrange=(0,
    1.5),
    axis=(xlabel=L"x", ylabel=L"y", title=L"|v(x,y) - \tilde{v}(x, y, +)|")
    )
    Colorbar(fig1[1,4], hm2)

    ax3, hm3 = heatmap(fig1[2, 1], kf_avgv,
    colormap=:balance,
    colorrange=(-maximum(kf_avg),
    maximum(kf_avg)),
    axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{v}(t = 30 \; \text{days}, x, y)")
    );
    Colorbar(fig1[2,2], hm3)
    ax4, hm4 = heatmap(fig1[2,3], abs.(states_true[end].v .- kf_avgv),
    colormap=:amp,
    colorrange=(0,
    1.5),
    axis=(xlabel=L"x", ylabel=L"y", title=L"|v(x,y) - \tilde{v}(x, y)|")
    )
    Colorbar(fig1[2,4], hm4)

    # energy plots

    fig1 = Figure(size=(600, 500));
    ax1, hm1 = heatmap(fig1[1,1], (states_true[end].u[:, 1:end-1].^2 .+ states_true[end].v[1:end-1, :].^2),
    colormap=:amp,
    axis=(xlabel=L"x", ylabel=L"y", title=L"\mathcal{E}"),
    colorrange=(0,
    maximum(states_true[end].u[:, 1:end-1].^2 .+ states_true[end].v[1:end-1, :].^2))
    );
    Colorbar(fig1[1,2], hm1)

    fig1 = Figure(size=(800, 700));
    ax1, hm1 = heatmap(fig1[1,1], states_adj[end].u[:, 1:end-1].^2 .+ states_adj[end].v[1:end-1, :].^2,
    colormap=:amp,
    axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{\mathcal{E}}(+)"),
    colorrange=(0,
    maximum(states_true[end].u[:, 1:end-1].^2 .+ states_true[end].v[1:end-1, :].^2)),
    );
    Colorbar(fig1[1,2], hm1)
    ax2, hm2 = heatmap(fig1[1,3], abs.((states_true[end].u[:, 1:end-1].^2 .+ states_true[end].v[1:end-1, :].^2) .- (states_adj[end].u[:, 1:end-1].^2 .+ states_adj[end].v[1:end-1, :].^2)),
    colormap=:amp,
    colorrange=(0,
    maximum(states_true[end].u[:, 1:end-1].^2 .+ states_true[end].v[1:end-1, :].^2)),
    axis=(xlabel=L"x", ylabel=L"y", title=L"|\mathcal{E} - \tilde{\mathcal{E}}(+)|")
    )
    Colorbar(fig1[1,4], hm2)


    ax3, hm3 = heatmap(fig1[2, 1], kf_avgu[:, 1:end-1].^2 .+ kf_avgv[1:end-1, :].^2,
    colormap=:amp,
    axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{\mathcal{E}}"),
    colorrange=(0,
    maximum(states_true[end].u[:, 1:end-1].^2 .+ states_true[end].v[1:end-1, :].^2)),
    );
    Colorbar(fig1[2,2], hm3)
    ax4, hm4 = heatmap(fig1[2,3], abs.((states_true[end].u[:, 1:end-1].^2 .+ states_true[end].v[1:end-1, :].^2) .- (kf_avgu[:, 1:end-1].^2 .+ kf_avgv[1:end-1, :].^2)),
    colormap=:amp,
    colorrange=(0,
    maximum(states_true[end].u[:, 1:end-1].^2 .+ states_true[end].v[1:end-1, :].^2)),
    axis=(xlabel=L"x", ylabel=L"y", title=L"|\mathcal{E} - \tilde{\mathcal{E}}|")
    )
    Colorbar(fig1[2,4], hm4)

    fig2 = Figure(size=(800, 500));

    ax1 = Axis(fig2[1,1])

    up_adj = zeros(65, 673)
    vp_adj = zeros(65, 673)

    up_ekf = zeros(65, 673)
    vp_ekf = zeros(65, 673)

    up_true = zeros(65, 673)
    vp_true = zeros(65, 673)
    for t = 1:673
        up_adj[:,t] = power(periodogram(states_adj[t].u; radialavg=true))
        vp_adj[:,t] = power(periodogram(states_adj[t].v; radialavg=true))

        up_ekf[:,t] = power(periodogram(ekf_avgu[t]; radialavg=true))
        vp_ekf[:,t] = power(periodogram(ekf_avgv[t]; radialavg=true))

        up_true[:,t] = power(periodogram(true_states[t].u; radialavg=true))
        vp_true[:,t] = power(periodogram(true_states[t].v; radialavg=true))
    end

    fftu_adj = fft(up_adj,[2])
    fftv_adj = fft(vp_adj,[2])
    adj_wl = 1 ./ freq(periodogram(states_adj[3].u; radialavg=true));
    adju_freq = LinRange(0, 672, 673)
    adju_freq = adju_freq ./ 673
    adju_freq = 1 ./ adju_freq 
    adju_freq[1] =  1000
    adj_wl[1] = 1000

    fftu_ekf = fft(up_ekf,[2])
    fftv_ekf = fft(vp_ekf,[2])
    ekf_wl = 1 ./ freq(periodogram(ekf_avgu[3]; radialavg=true));
    ekfu_freq = LinRange(0, 672, 673)
    ekfu_freq = ekfu_freq ./ 673
    ekfu_freq = 1 ./ ekfu_freq 
    ekfu_freq[1] =  1000
    ekf_wl[1] = 1000

    fftu_true = fft(up_ekf,[2])
    fftv_true = fft(vp_ekf,[2])
    true_wl = 1 ./ freq(periodogram(true_states[3].u; radialavg=true));
    trueu_freq = LinRange(0, 672, 673)
    trueu_freq = trueu_freq ./ 673
    trueu_freq = 1 ./ trueu_freq 
    trueu_freq[1] =  1000
    true_wl[1] = 1000

    heatmap(adj_wl, adju_freq, abs.(fftu_adj) + abs.(fftv_adj); colorscale=log10, axis=(xscale=log10, yscale=log10))

    # one dimensional figures
    fig2 = Figure(size=(800, 500));
    ax = Axis(fig2)
    t = 10
    lines(fig2[1,1], adj_wl[2:end], up_adj[2:end,t] + vp_adj[2:end,t], label="Adjoint", axis=(xscale=log10,yscale=log10,xlabel="Wavelength", ylabel="KE(k)"))
    lines!(fig2[1,1], ekf_wl[2:end], up_ekf[2:end,t] + vp_ekf[2:end,t], label="EKF")
    lines!(fig2[1,1], true_wl[2:end], up_true[2:end,t] + vp_true[2:end,t], label="Truth")
    axislegend()

    fig2 = Figure(size=(800, 500));

    t = 1
    lines(fig2[1,1], adj_wl[2:end], abs.(fft(up_adj[2:end,t])) + abs.(fft(vp_adj[2:end,t])), axis=(xscale=log10,yscale=log10))
    lines!(fig2[1,1], ekf_wl[2:end], abs.(fft(up_ekf[2:end,t])) + abs.(fft(vp_ekf[2:end,t])))
    lines!(fig2[1,1], true_wl[2:end], abs.(fft(up_true[2:end,t])) + abs.(fft(vp_true[2:end,t])))

    # two dimensional freq power plot

    adjmatu = zeros(65, 673)
    adjmatv = zeros(65, 673)

    for j = 1:673
        adjmatu[:, j] = adjfreqpoweru[j].power
        adjmatv[:, j] = adjfreqpowerv[j].power
    end


end

########################################################################
# Derivative check

# function finite_difference_only(data_spots,x_coord,y_coord;kwargs...)

#     data, true_states = generate_data(data_spots, sigma_data; kwargs...)
#     P1 = ShallowWaters.Parameter(T=Float32;kwargs...)

#     S_outer = ShallowWaters.model_setup(P1)

#     dS_outer = Enzyme.Compiler.make_zero(S_outer)

#     ddata = Enzyme.make_zero(data)
#     ddata_spots = Enzyme.make_zero(data_spots)

#     autodiff(Enzyme.ReverseWithPrimal, integrate,
#         Duplicated(S_outer, dS_outer),
#         Duplicated(data,ddata),
#         Duplicated(data_spots, ddata_spots)
#     )

#     enzyme_deriv = dS_outer.Prog.u[x_coord, y_coord]

#     steps = [50, 40, 30, 20, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

#     P2 = ShallowWaters.Parameter(T=Float32;kwargs...)
#     S = ShallowWaters.model_setup(P2)
#     J_outer = integrate(S, data, data_spots)

#     diffs = []

#     for s in steps

#         P = ShallowWaters.Parameter(T=Float32;kwargs...)
#         S_inner = ShallowWaters.model_setup(P)

#         S_inner.Prog.u[x_coord, y_coord] += s

#         # J_inner = checkpointed_integration(S_inner, revolve)
#         J_inner = integrate(S_inner,data,data_spots)

#         push!(diffs, (J_inner - J_outer) / s)

#     end

#     return diffs, enzyme_deriv

# end

# diffs, enzyme_deriv = finite_difference_only(data_spots, 72, 64;
#     output=false,
#     L_ratio=1,
#     g=9.81,
#     H=500,
#     wind_forcing_x="double_gyre",
#     Lx=3840e3,
#     tracer_advection=false,
#     tracer_relaxation=false,
#     seasonal_wind_x=false,
#     data_steps=data_steps,
#     topography="flat",
#     bc="nonperiodic",
#     α=2,
#     nx=128,
#     Ndays=Ndays,
#     initial_cond="ncfile",
#     initpath="./data_files_forkf/128_spinup_noforcing/"
# )