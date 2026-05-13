model_data = deepcopy(ekf_model);
model_nodata = deepcopy(model_data);
model_nodata.data_steps = 3000:3000:3000;

create_ensemble(model_data, model_data.N, param_guess)
create_ensemble(model_nodata, model_nodata.N, param_guess)

ekf_avgudata, ekf_avgvdata, ekf_avgetadata = integrate_ensemble(model_data, 10*224);
ekf_avgunodata, ekf_avgvnodata, ekf_avgetanodata = integrate_ensemble(model_nodata, 10*224);

Progkf_nodata = []
Progkf_data = []

S_all_data = model_data.enkfvars.S_ensemble;
S_all_nodata = model_nodata.enkfvars.S_ensemble;

for n = 1:model_data.N
    
    push!(Progkf_data, ShallowWaters.PrognosticVars{Float64}(ShallowWaters.remove_halo(
        S_all_data[n].Prog.u,
        S_all_data[n].Prog.v,
        S_all_data[n].Prog.η,
        S_all_data[n].Prog.sst,
        S_all_data[n]
    )...)
    )

    push!(Progkf_nodata, ShallowWaters.PrognosticVars{Float64}(ShallowWaters.remove_halo(
        S_all_nodata[n].Prog.u,
        S_all_nodata[n].Prog.v,
        S_all_nodata[n].Prog.η,
        S_all_nodata[n].Prog.sst,
        S_all_nodata[n]
    )...)
    )

end

fig = Figure(size=(550, 200));

n = 2
ax = Axis(fig[1,1], title="True - no data ensemble member")
hm = heatmap!(ax, ud[:,:,2] .- Progkf_nodata[n].u)
Colorbar(fig[1,2], hm)
ax2 = Axis(fig[1,3], title="Data ensemble - no data ensemble")
hm2 = heatmap!(ax2,Progkf_data[n].u .- Progkf_nodata[n].u)
Colorbar(fig[1,4], hm2)

fig = Figure(size=(550, 200));

n = 2
ax = Axis(fig[1,1], title="True - no data ensemble member")
hm = heatmap!(ax, ud[:,:,2] .- ekf_avgunodata[end])
Colorbar(fig[1,2], hm)
ax2 = Axis(fig[1,3], title="Data ensemble - no data ensemble")
hm2 = heatmap!(ax2,ekf_avgudata[end] .- ekf_avgunodata[end])
Colorbar(fig[1,4], hm2)

fig = Figure(size=(550, 200));

ax = Axis(fig[1,1], title="True - avg. no data ensemble member")
hm = heatmap!(ax, ud[:,:,2] .- sum(Progkf_nodata[j].u for j in 1:model_data.N) ./ 100)
Colorbar(fig[1,2], hm)
ax2 = Axis(fig[1,3], title="Avg. Data ensemble - avg. no data ensemble")
hm2 = heatmap!(ax2, sum(Progkf_nodata[j].u for j in 1:model_data.N) ./ 100
    .- sum(Progkf_data[j].u for j in 1:model_data.N) ./ 100
)
Colorbar(fig[1,4], hm2)

fig = Figure(size=(700, 200));

ax0 = Axis(fig[1,1], title="True state")
hm0 = heatmap!(ax0, ud[:,:,2])
Colorbar(fig[1,2], hm0)
ax = Axis(fig[1,3], title="Predicted state")
hm = heatmap!(ax, states_pred[end].u)
Colorbar(fig[1,4], hm)
ax2 = Axis(fig[1,5], title="Data ensemble average")
hm2 = heatmap!(ax2, ekf_avgudata[end])#sum(Progkf_data[j].u for j in 1:model_data.N) ./ 100)
Colorbar(fig[1,6], hm2)

ud = ncread("./data_files/128_90days_postspinup_hourlysaves/u.nc", "u");
vd = ncread("./data_files/128_90days_postspinup_hourlysaves/v.nc", "v");
etad = ncread("./data_files/128_90days_postspinup_hourlysaves/eta.nc", "eta");

n = 24
true_energy = zeros(n)
pred_energy = zeros(n)
ekf_energy = zeros(n)

for t = 1:n

    true_energy[t] = (sum(ud[:,:,t].^2) + sum(vd[:,:,t].^2)) / (128 * 127)
    pred_energy[t] = (sum(states_pred[t+1].u.^2) + sum(states_pred[t+1].v.^2)) / (128 * 127)
    ekf_energy[t] = (sum(ekf_avgudata[t].^2) + sum(ekf_avgvdata[t].^2)) / (128 * 127)

end

fig = Figure(size=(800, 700));
ax = Axis(fig[1,1])
lines!(ax, LinRange(0, 225*Ndays, 24), true_energy[1:24], label="Truth")
lines!(ax,LinRange(0, 225*Ndays, n), pred_energy, label="Prediction",linestyle=:dash)
lines!(ax,LinRange(0, 225*Ndays, n), ekf_energy, label="EKF")
vlines!(ax, data_steps, color=:gray75, linestyle=:dot);
axislegend()

fig = Figure();
ax = Axis(fig[1,1])
hm = heatmap!(ax, abs.(sum(Progkf_nodata[j].u for j in 1:model_data.N) ./ 100 .- states_pred[end].u),colormap=:amp)
Colorbar(fig[1,2], hm)
ax2 = Axis(fig[1,3])
hm2 = heatmap!(ax2, abs.(ekf_avgunodata[end] .- states_pred[end].u),colormap=:amp)
Colorbar(fig[1,4], hm2)
