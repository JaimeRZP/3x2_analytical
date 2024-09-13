using LinearAlgebra
using Turing
using LimberJack
using CSV
using DataFrames
using YAML
using NPZ
using JLD2
using PythonCall
using Statistics
sacc = pyimport("sacc");


method = "bpz"
sacc_path = "../../data/CosmoDC2/summary_statistics_fourier_tjpcov.sacc"
yaml_path = "../../data/CosmoDC2/wlwl.yml"
nz_path = string("../../data/CosmoDC2/image_nzs_", method, "_priors/")
dz_path = string("../../data/CosmoDC2/image_dz_", method, "_priors/dz_prior.npz")
fake_data_path = string("../../data/CosmoDC2/CosmoDC2_wlwl_theory_photo_", method, "_best.csv")

fake_data = CSV.read(fake_data_path, DataFrame)
fake_data = fake_data.theory[1:end-1]

sacc_file = sacc.Sacc().load_fits(sacc_path)
yaml_file = YAML.load_file(yaml_path)

nz_source_0 = npzread(string(nz_path, "nz_source_0.npz"))
nz_source_1 = npzread(string(nz_path, "nz_source_1.npz"))
nz_source_2 = npzread(string(nz_path, "nz_source_2.npz"))
nz_source_3 = npzread(string(nz_path, "nz_source_3.npz"))
nz_source_4 = npzread(string(nz_path, "nz_source_4.npz"))
zs_k0, nz_k0 = nz_source_0["z"], nz_source_0["dndz"]
zs_k1, nz_k1 = nz_source_1["z"], nz_source_1["dndz"]
zs_k2, nz_k2 = nz_source_2["z"], nz_source_2["dndz"]
zs_k3, nz_k3 = nz_source_3["z"], nz_source_3["dndz"]
zs_k4, nz_k4 = nz_source_4["z"], nz_source_4["dndz"]
mu_k0 = sum(zs_k0 .* nz_k0) / sum(nz_k0)
mu_k1 = sum(zs_k1 .* nz_k1) / sum(nz_k1)
mu_k2 = sum(zs_k2 .* nz_k2) / sum(nz_k2)
mu_k3 = sum(zs_k3 .* nz_k3) / sum(nz_k3)
mu_k4 = sum(zs_k4 .* nz_k4) / sum(nz_k4)

dz_prior = npzread(dz_path)
dz_mean, dz_cov = dz_prior["mean"], dz_prior["cov"]
dz_mean = dz_mean[1:10]
dz_cov = dz_cov[1:10, 1:10]
dz_chol = cholesky(dz_cov).U'

meta, files = make_data(sacc_file, yaml_file;
                        nz_source_0=nz_source_0,
                        nz_source_1=nz_source_1,
                        nz_source_2=nz_source_2,
                        nz_source_3=nz_source_3,
                        nz_source_4=nz_source_4)

meta.types = [    
    "galaxy_shear", 
    "galaxy_shear", 
    "galaxy_shear",
    "galaxy_shear",
    "galaxy_shear"]

data = fake_data
cov = meta.cov

Γ = sqrt(cov)
iΓ = inv(Γ)
data = iΓ * data

init_params=[0.30, 0.5, 0.67, 0.81, 0.95]

@model function model(data;
    meta=meta, 
    files=files)

    #KiDS priors
    Ωm ~ Uniform(0.2, 0.6)
    Ωbb ~ Uniform(0.28, 0.65) # 10*Ωb 
    Ωb := 0.1*Ωbb 
    h ~ Truncated(Normal(0.72, 0.05), 0.64, 0.82)
    σ8 ~ Uniform(0.4, 1.2)
    ns ~ Uniform(0.84, 1.1)

    alphas ~ filldist(Normal(0, 1), 10)
    SnWs = dz_mean .+ dz_chol * alphas
    dzs := [SnWs[1], SnWs[3], SnWs[5], SnWs[7], SnWs[9]]
    wzs := [SnWs[2], SnWs[4], SnWs[6], SnWs[8], SnWs[10]]
    source_0_zs = ((zs_k0-mu_k0)/wzs[1] + mu_k0 + dzs[1])
    source_1_zs = ((zs_k1-mu_k1)/wzs[2] + mu_k1 + dzs[2])
    source_2_zs = ((zs_k2-mu_k2)/wzs[3] + mu_k2 + dzs[3])
    source_3_zs = ((zs_k3-mu_k3)/wzs[4] + mu_k3 + dzs[4])
    source_4_zs = ((zs_k4-mu_k4)/wzs[5] + mu_k4 + dzs[5])

    nuisances = Dict{String, Float64}(
        "source_0_b"    => 0.879118,
        "source_1_b"    => 1.05894,
        "source_2_b"    => 1.22145,
        "source_3_b"    => 1.35065,
        "source_4_b"    => 1.58909,
        "source_0_zs"   => source_0_zs,
        "source_1_zs"   => source_1_zs,
        "source_2_zs"   => source_2_zs,
        "source_3_zs"   => source_3_zs,
        "source_4_zs"   => source_4_zs,
        "source_0_m"  => -0.00733846,
        "source_1_m"  => -0.00434667,
        "source_2_m"  => 0.00434908,
        "source_3_m"  => -0.00278755,
        "source_4_m"  => 0.000101118)
        
    cosmology = Cosmology(Ωm=Ωm, Ωb=Ωb, h=h, ns=ns, σ8=σ8,
            tk_mode=:EisHu,
            pk_mode=:Halofit)

    theory := Theory(cosmology, meta, files; Nuisances=nuisances)
    data ~ MvNormal(iΓ * theory, I)
end

iterations = 2000
adaptation = 500
TAP = 0.65

println("sampling settings: ")
println("iterations ", iterations)
println("TAP ", TAP)
println("adaptation ", adaptation)
#println("nchains ", nchains)

# Start sampling.
folpath = "../../chains_right_nzs/numerical/"
folname = string("CosmoDC2_wlwl_dz_num_TAP_", TAP)
folname = joinpath(folpath, folname)

if isdir(folname)
    fol_files = readdir(folname)
    println("Found existing file ", folname)
    if length(fol_files) != 0
        last_chain = last([file for file in fol_files if occursin("chain", file)])
        last_n = parse(Int, last_chain[7])
        #println("Restarting chain")
    else
        #println("Starting new chain")
        last_n = 0
    end
else
    mkdir(folname)
    println(string("Created new folder ", folname))
    last_n = 0
end

# Create a placeholder chain file.
CSV.write(joinpath(folname, string("chain_", last_n+1,".csv")), Dict("params"=>[]), append=true)

# Sample
cond_model = model(data)
sampler = NUTS(adaptation, TAP)
chain = sample(cond_model, sampler, iterations;
                init_params=init_params,
                progress=true, save_state=true)

# Save the actual chain.       
@save joinpath(folname, string("chain_", last_n+1,".jls")) chain
CSV.write(joinpath(folname, string("chain_", last_n+1,".csv")), chain)
CSV.write(joinpath(folname, string("summary_", last_n+1,".csv")), describe(chain)[1])
