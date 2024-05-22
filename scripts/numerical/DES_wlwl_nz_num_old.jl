using LinearAlgebra
using Turing
using MyLimberJack
using CSV
using YAML
using JLD2
using NPZ
using PythonCall
sacc = pyimport("sacc");

#println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

sacc_path = "../../data/FD/cls_FD_covG.fits"
yaml_path = "../../data/DESY1/wlwl.yml"
nz_path = "../../data/DESY1/nzs/"
sacc_file = sacc.Sacc().load_fits(sacc_path)
yaml_file = YAML.load_file(yaml_path)
nz_DESwl__0 = npzread(string(nz_path, "nz_DESwl__0.npz"))
nz_DESwl__1 = npzread(string(nz_path, "nz_DESwl__1.npz"))
nz_DESwl__2 = npzread(string(nz_path, "nz_DESwl__2.npz"))
nz_DESwl__3 = npzread(string(nz_path, "nz_DESwl__3.npz"))
zs_k0, nz_k0, cov_k0 = nz_DESwl__0["z"], nz_DESwl__0["dndz"], nz_DESwl__0["cov"]
zs_k1, nz_k1, cov_k1 = nz_DESwl__1["z"], nz_DESwl__1["dndz"], nz_DESwl__1["cov"]
zs_k2, nz_k2, cov_k2 = nz_DESwl__2["z"], nz_DESwl__2["dndz"], nz_DESwl__2["cov"]
zs_k3, nz_k3, cov_k3 = nz_DESwl__3["z"], nz_DESwl__3["dndz"], nz_DESwl__3["cov"]
chol_k0 = cholesky(cov_k0).U'
chol_k1 = cholesky(cov_k1).U'
chol_k2 = cholesky(cov_k2).U'
chol_k3 = cholesky(cov_k3).U'
meta, files = make_data(sacc_file, yaml_file;
                        nz_DESwl__0=nz_DESwl__0,
                        nz_DESwl__1=nz_DESwl__1,
                        nz_DESwl__2=nz_DESwl__2,
                        nz_DESwl__3=nz_DESwl__3);

data = meta.data
cov = meta.cov

Γ = sqrt(cov)
iΓ = inv(Γ)
data = iΓ * data

init_params=[0.30, 0.05, 0.67, 0.81, 0.95]
init_params = [init_params; 
    zeros(length(zs_k0));
    zeros(length(zs_k1));
    zeros(length(zs_k2));
    zeros(length(zs_k3))]

@model function model(data;
    meta=meta, 
    files=files)

    Ωm ~ Uniform(0.2, 0.6)
    s8 ~ Uniform(0.6, 0.9)
    Ωb ~ Uniform(0.03, 0.07)
    h ~ Uniform(0.55, 0.91)
    ns ~ Uniform(0.87, 1.07)

    cosmology = Cosmology(Ωm, Ωb, h, ns, s8,
                        tk_mode="EisHu",
                        Pk_mode="Halofit")

    A_IA = 0.0 #~ Uniform(-5, 5)
    alpha_IA = 0.0 #~ Uniform(-5, 5)

    n = length(nz_k0)
    DESwl__0_nz = zeros(cosmology.settings.cosmo_type, n)
    DESwl__1_nz = zeros(cosmology.settings.cosmo_type, n)
    DESwl__2_nz = zeros(cosmology.settings.cosmo_type, n)
    DESwl__3_nz = zeros(cosmology.settings.cosmo_type, n)
    for i in 1:n
        DESwl__0_nz[i] ~ TruncatedNormal(nz_k0[i], sqrt.(diag(cov_k0))[i], -0.07, 0.5)
        DESwl__1_nz[i] ~ TruncatedNormal(nz_k1[i], sqrt.(diag(cov_k1))[i], -0.07, 0.5)
        DESwl__2_nz[i] ~ TruncatedNormal(nz_k2[i], sqrt.(diag(cov_k2))[i], -0.07, 0.5)
        DESwl__3_nz[i] ~ TruncatedNormal(nz_k3[i], sqrt.(diag(cov_k3))[i], -0.07, 0.5)
    end

    DESwl__0_m = 0.012 #~ Normal(0.012, 0.023)
    DESwl__1_m = 0.012 #~ Normal(0.012, 0.023)
    DESwl__2_m = 0.012 #~ Normal(0.012, 0.023)
    DESwl__3_m = 0.012 #~ Normal(0.012, 0.023)

    nuisances = Dict("A_IA" => A_IA,
                     "alpha_IA" => alpha_IA,
                     "DESwl__0_nz" => DESwl__0_nz,
                     "DESwl__1_nz" => DESwl__1_nz,
                     "DESwl__2_nz" => DESwl__2_nz,
                     "DESwl__3_nz" => DESwl__3_nz,
                     "DESwl__0_m" => DESwl__0_m,
                     "DESwl__1_m" => DESwl__1_m,
                     "DESwl__2_m" => DESwl__2_m,
                     "DESwl__3_m" => DESwl__3_m)


    theory = Theory(cosmology, meta, files; Nuisances=nuisances)
    data ~ MvNormal(iΓ * theory, I)
end

iterations = 2000
adaptation = 500
TAP = 0.65
init_ϵ = 0.03

println("sampling settings: ")
println("iterations ", iterations)
println("TAP ", TAP)
println("adaptation ", adaptation)
#println("nchains ", nchains)

# Start sampling.
folpath = "../../chains/numerical/"
folname = string("DES_wlwl_nz_num_old_TAP_", TAP,  "_init_ϵ_", init_ϵ)
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
sampler = NUTS(adaptation, TAP; init_ϵ=init_ϵ)
chain = sample(cond_model, sampler, iterations;
                init_params=init_params,
                progress=true, save_state=true)

# Save the actual chain.       
@save joinpath(folname, string("chain_", last_n+1,".jls")) chain
CSV.write(joinpath(folname, string("chain_", last_n+1,".csv")), chain)
CSV.write(joinpath(folname, string("summary_", last_n+1,".csv")), describe(chain)[1])
