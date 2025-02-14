function exp1_cpintegrate(S, scheme, data, data_spots)

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

    # run integration loop with checkpointing
    j = 1

    @checkpoint_struct scheme S for S.parameters.i = 1:S.grid.nt

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

            tempu = u_mat_to_vec(temp.u)[data_spots]

            S.parameters.J += sum((tempu - data[:, j]).^2)

            j += 1

        end

        # Copy back from substeps
        copyto!(u,u0)
        copyto!(v,v0)
        copyto!(η,η0)

    end

    return nothing

    return S.parameters.J

end

function exp1_integrate(S, data, data_spots)

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

            tempu = u_mat_to_vec(temp.u)[data_spots]

            S.parameters.J = S.parameters.J + sum((tempu - data[:, j]).^2)

            j += 1

        end

        # Copy back from substeps
        copyto!(u,u0)
        copyto!(v,v0)
        copyto!(η,η0)

    end

    @show S.parameters.J
    return S.parameters.J

end

function exp1_cost_eval(param_guess, data, data_spots, data_steps, Ndays)

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
    α=2,
    nx=128,
    Ndays=Ndays,
    initial_cond="ncfile",
    initpath="./data_files_forkf/128_spinup_noforcing/"
    )
    S = ShallowWaters.model_setup(P)

    S.Prog.u = reshape(param_guess[1:17292], 131, 132)
    S.Prog.v = reshape(param_guess[17293:34584], 132, 131)
    S.Prog.η = reshape(param_guess[34585:end], 130, 130)

    J = exp1_integrate(S, data, data_spots)

    return J

end

function exp1_gradient_eval(G, param_guess, data, data_spots, data_steps, Ndays)

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
    α=2,
    nx=128,
    Ndays=Ndays,
    initial_cond="ncfile",
    initpath="./data_files_forkf/128_spinup_noforcing/"
    )
    S = ShallowWaters.model_setup(P)

    m,n = size(S.Diag.VolumeFluxes.h)
    @show m
    @show n

    S.Prog.u = reshape(param_guess[1:17292], 131, 132)
    S.Prog.v = reshape(param_guess[17293:34584], 132, 131)
    S.Prog.η = reshape(param_guess[34585:end], 130, 130)

    dS = Enzyme.Compiler.make_zero(S)
    dS.parameters.J = 1.0
    snaps = Int(floor(sqrt(S.grid.nt)))
    revolve = Revolve{ShallowWaters.ModelSetup}(S.grid.nt,
        snaps;
        verbose=1,
        gc=true,
        write_checkpoints=false,
        write_checkpoints_filename = "",
        write_checkpoints_period = 224
    )

    ddata = Enzyme.make_zero(data)
    ddata_spots = Enzyme.make_zero(data_spots)

    autodiff(Enzyme.ReverseWithPrimal, exp1_integrate,
    Duplicated(S, dS),
    Duplicated(data, ddata),
    Duplicated(data_spots, ddata_spots)
    )

    # @show dS.Prog.u[5:7,50:60]
    G .= [vec(dS.Prog.u); vec(dS.Prog.v); vec(dS.Prog.η)]
    # @show G[5*128:5*128+10]

    return nothing

end

function exp1_FG(F, G, param_guess, data, data_spots, data_steps, Ndays)

    G === nothing || exp1_gradient_eval(G, param_guess, data, data_spots, data_steps, Ndays)
    F === nothing || return exp1_cost_eval(param_guess, data, data_spots, data_steps, Ndays)

end

function exp1_initialcond(N, data_spots, sigma_initcond, sigma_data; kwargs...)

    P_pred = ShallowWaters.Parameter(T=Float32;kwargs...)

    S_true = ShallowWaters.model_setup(P_pred)

    data, true_states = generate_data(S_true, data_spots, sigma_data)

    S_pred = ShallowWaters.model_setup(P_pred)

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

    param_guess = [vec(uic); vec(vic); vec(etaic)]

    S_kf_all, Progkf_all = run_ensemble_kf(N,
    data,
    param_guess,
    data_spots,
    sigma_initcond,
    sigma_data;
    kwargs...
    )

    dS = Enzyme.Compiler.make_zero(S_pred)
    # temp = [vec(dS.Prog.u);vec(dS.Prog.v); vec(dS.Prog.η)]

    G = zeros(length(dS.Prog.u) + length(dS.Prog.v) + length(dS.Prog.η))

    # exp1_gradient_eval(G, param_guess, data, data_spots)

    Ndays = copy(P_pred.Ndays)
    data_steps = copy(P_pred.data_steps)

    fg!_closure(F, G, ic) = exp1_FG(F, G, ic, data, data_spots, data_steps, Ndays)
    obj_fg = Optim.only_fg!(fg!_closure)
    result = Optim.optimize(obj_fg, param_guess, Optim.LBFGS(), Optim.Options(show_trace=true, iterations=1))

    S_adj = ShallowWaters.model_setup(P_pred)
    S_adj.Prog.u = reshape(result.minimizer[1:17292], 131, 132)
    S_adj.Prog.v = reshape(result.minimizer[17293:34584], 132, 131)
    S_adj.Prog.η = reshape(result.minimizer[34585:end], 130, 130)
    _, states_adj = generate_data(S_adj, data_spots, sigma_data)

    return S_kf_all, Progkf_all, G, dS, data, true_states, result, S_adj, states_adj

end

function run_exp1()

    N = 3
    sigma_data = 0.01
    sigma_initcond = 0.01
    data_steps = 1:1:6733
    data_spots = 5:100:128*127
    Ndays = 10

    S_kf_all, Progkf_all, G, dS, data, states_true, result, S_adj, states_adj = exp1_initialcond(N,
        data_spots,
        sigma_initcond,
        sigma_data,
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
        α=2,
        nx=128,
        Ndays=Ndays,
        initial_cond="ncfile",
        initpath="./data_files_forkf/128_spinup_noforcing/"
    )

    return S_kf_all, Progkf_all, G, dS, data, states_true, result, S_adj, states_adj

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