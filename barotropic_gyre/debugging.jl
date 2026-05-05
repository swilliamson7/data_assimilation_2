model_data = deepcopy(ekf_model);
model_nodata = deepcopy(model_data);
model_nodata.data_steps = 3000:3000:3000;

create_ensemble(model_data, model_data.N, param_guess)
create_ensemble(model_nodata, model_nodata.N, param_guess)

integrate_ensemble(model_data, 10*224);
integrate_ensemble(model_nodata, 10*224);

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

n = 10
ax = Axis(fig[1,1], title="True - no data ensemble member")
hm = heatmap!(ax, ud[:,:,10] .- Progkf_nodata[n].u)
Colorbar(fig[1,2], hm)
ax2 = Axis(fig[1,3], title="Data ensemble - no data ensemble")
hm2 = heatmap!(ax2,Progkf_data[n].u .- Progkf_nodata[n].u)
Colorbar(fig[1,4], hm2)

fig = Figure(size=(550, 200));

ax = Axis(fig[1,1], title="True - no data ensemble member")
hm = heatmap!(ax, abs.(ud[:,:,10] .- sum(Progkf_nodata[j].u for j in 1:model_data.N) ./ 100))
Colorbar(fig[1,2], hm)
ax2 = Axis(fig[1,3], title="Data ensemble - no data ensemble")
hm2 = heatmap!(ax2,abs.(sum(Progkf_nodata[j].u for j in 1:model_data.N) ./ 100
    .- sum(Progkf_data[j].u for j in 1:model_data.N) ./ 100)
)
Colorbar(fig[1,4], hm2)

fig = Figure(size=(550, 200));

ax0 = Axis(fig[1,1], title="True state")
hm0 = heatmap!(ax0, ud[:,:,10])
Colorbar(fig[1,2], hm0)
ax = Axis(fig[1,3], title="Predicted state")
hm = heatmap!(ax, states_pred[end].u)
Colorbar(fig[1,4], hm)
ax2 = Axis(fig[1,5], title="Data ensemble average")
hm2 = heatmap!(ax2, sum(Progkf_data[j].u for j in 1:model_data.N) ./ 100)
Colorbar(fig[1,6], hm2)