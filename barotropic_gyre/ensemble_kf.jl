# We need a version of the integration function that runs a single timestep and 
# takes as input the current prognostic fields. This function will then be
# passed to Enzyme for computing the Jacobian, which can subsequently be used
# in the Kalman filter. We're giving as input uveta, which will be a block
# vector of the fields u, v, and eta in that order. The restructuring of the
# arrays will be as columns stacked on top of eachother, e.g. the first column becomes
# the first bit of the vector, the second column the second bit, and so on.

using Parameters
Enzyme.API.looseTypeAnalysis!(true)

function u_mat_to_vec(u_mat)

    m, n = size(u_mat)
    u_vec = reshape(u_mat, (m*n))

    return u_vec

end

function v_mat_to_vec(v_mat)

    m, n = size(v_mat)
    v_vec = reshape(v_mat, (m*n))

    return v_vec

end

function eta_mat_to_vec(eta_mat)

    m, n = size(eta_mat)
    eta_vec = reshape(eta_mat, (m*n))

    return eta_vec

end

function u_vec_to_mat(u_vec,S)
    nx = S.grid.nx
    u_mat = reshape(u_vec, nx-1, nx)
    return u_mat
end

function v_vec_to_mat(v_vec,S)
    nx = S.grid.nx
    v_mat = reshape(v_vec, nx, nx-1)
    return v_mat
end

function eta_vec_to_mat(eta_vec,S)
    nx = S.grid.nx
    eta_mat = reshape(eta_vec,nx,nx)
    return eta_mat
end

function one_step_function(S)

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

    # Copy back from substeps
    copyto!(u,u0)
    copyto!(v,v0)
    copyto!(η,η0)

    return ShallowWaters.PrognosticVars{S.parameters.Tprog}(ShallowWaters.remove_halo(u,v,η,sst,S)...)

end

# This function will setup the ensemble Kalman filter method for the barotropic gyre. It takes as input:
# N - the number of ensembles to create
# Ndays - how long to run the intergration/data assimilation process for
function generate_data(Ndays)

    # Create the "true" model, from this we'll make data
    S_true = ShallowWaters.model_setup(
        output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        seasonal_wind_x=false,
        tracer_advection=false,
        tracer_relaxation=false,
        topography="flat",
        bc="nonperiodic",
        α=2,
        nx=128,
        Ndays = Ndays,
        initial_cond="ncfile",
        initpath="./data_files_forkf/128_spinup_noforcing/"
    )

    data = zeros(127*128, S_true.grid.nt)
    j = 1

    for t = 1:S_true.grid.nt

        P = one_step_function(S_true)

        data[:, j] = u_mat_to_vec(P.u)

    end
    
    # add noise to the data
    for k = 1:S_true.grid.nt
        data[:, k] = data[:, k] .+ randn(127*128)
    end

    return data

end

function run_ensemble_kf(N, Ndays)

    Π = (I - (1 / N)*(ones(N) * ones(N)')) / sqrt(N - 1)
    W = zeros(N,N)
    T = zeros(N,N)

    data = generate_data(Ndays)
    data_steps = 1:100:6733
    S = zeros(length(data[:,1]), N)
    U = zeros(length(data[:,1]), N)

    # Generate the initial model realization ensemble,
    # generated by slightly perturbing the initial condition N times. 
    # Output will be stored in the matrix Z, and all model structs will be 
    # stored in S_all
    Z = zeros(48896, N)
    S_all = []
    for n = 1:N

        S_kf = ShallowWaters.model_setup(
            output=false,
            L_ratio=1,
            g=9.81,
            H=500,
            wind_forcing_x="double_gyre",
            Lx=3840e3,
            tracer_advection=false,
            tracer_relaxation=false,
            seasonal_wind_x=false,
            topography="flat",
            bc="nonperiodic",
            α=2,
            nx=128,
            Ndays = Ndays,
            initial_cond="ncfile",
            initpath="./data_files_forkf/128_spinup_noforcing/"
        )

        # perturb initial conditions from those seen by the KF(create incorrect initial conditions)
        S_kf.Prog.u[3:end-2, :] = S_kf.Prog.u[3:end-2, :] + randn(size(S_kf.Prog.u[3:end-2, :]))
        S_kf.Prog.v[:, 3:end-2] = S_kf.Prog.v[:, 3:end-2] + randn(size(S_kf.Prog.v[:, 3:end-2]))
        S_kf.Prog.η = S_kf.Prog.η + randn(size(S_kf.Prog.η))

        push!(S_all, S_kf)

        temp = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(S_kf.Prog.u,
            S_kf.Prog.v,
            S_kf.Prog.η,
            S_kf.Prog.sst,S_kf)...
        )

        Z[:, n] = [u_mat_to_vec(temp.u); v_mat_to_vec(temp.v); eta_mat_to_vec(temp.η)]

    end

    for t = 1:S_kf.grid.nt

        P_all = []

        for n = 1:N

            p = one_step_function(S_all[n])
            push!(P_all, p)

        end

        if t ∈ data_steps

            d = data[:, t]
            E = randn(127*128, N)
            D = d * ones(N)' + sqrt(N - 1) * E
            E = D * Π

            for k = 1:N

                Z[:, k] = [u_mat_to_vec(P_all[k].u); v_mat_to_vec(P_all[k].v); eta_mat_to_vec(P_all[k].η)]
                U[:, k] = Z[1:127*128, k]

            end

            Y = U * Π

            S .= Y
            D̃ = D - U
            W = S' * (S*S' + E*E')^(-1)*D̃

            T = (I + W./(sqrt(N-1)))

            Z = Z*T

            for k = 1:N

                # u_mat = u_vec_to_mat(Z[1:127*128, k], S_all[k])
                # v_mat = v_vec_to_mat(Z[127*128+1:(2*127*128),k],S_all[k])
                # eta_mat = eta_vec_to_mat(Z[(2*127*128)+1:end,k], S_all[k])

                u,v,eta = ShallowWaters.add_halo(P_all[k].u,P_all[k].v,P_all[k].η,S_all[k])

                S_all[k].Prog.u = u
                S_all[k].Prog.v = v 
                S_all[k].Prog.η = eta

            end

        end

    end

    return S_all

end