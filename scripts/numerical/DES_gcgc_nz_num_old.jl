using LinearAlgebra
using Turing
using LimberJack
using CSV
using YAML
using NPZ
using JLD2
using PythonCall
sacc = pyimport("sacc");

#println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

sacc_path = "../../data/FD/cls_FD_covG.fits"
yaml_path = "../../data/DESY1/gcgc.yml"
nz_path = "../../data/DESY1/nzs/"
sacc_file = sacc.Sacc().load_fits(sacc_path)
yaml_file = YAML.load_file(yaml_path)
nz_DESgc__0 = npzread(string(nz_path, "nz_DESgc__0.npz"))
nz_DESgc__1 = npzread(string(nz_path, "nz_DESgc__1.npz"))
nz_DESgc__2 = npzread(string(nz_path, "nz_DESgc__2.npz"))
nz_DESgc__3 = npzread(string(nz_path, "nz_DESgc__3.npz"))
nz_DESgc__4 = npzread(string(nz_path, "nz_DESgc__4.npz"))
zs_k0, nz_k0, cov_k0 = nz_DESgc__0["z"], nz_DESgc__0["dndz"], nz_DESgc__0["cov"]
zs_k1, nz_k1, cov_k1 = nz_DESgc__1["z"], nz_DESgc__1["dndz"], nz_DESgc__1["cov"]
zs_k2, nz_k2, cov_k2 = nz_DESgc__2["z"], nz_DESgc__2["dndz"], nz_DESgc__2["cov"]
zs_k3, nz_k3, cov_k3 = nz_DESgc__3["z"], nz_DESgc__3["dndz"], nz_DESgc__3["cov"]
zs_k4, nz_k4, cov_k4 = nz_DESgc__4["z"], nz_DESgc__4["dndz"], nz_DESgc__4["cov"]
chol_k0 = cholesky(cov_k0).U'
chol_k1 = cholesky(cov_k1).U'
chol_k2 = cholesky(cov_k2).U'
chol_k3 = cholesky(cov_k3).U'
chol_k4 = cholesky(cov_k4).U'
meta, files = make_data(sacc_file, yaml_file;
                        nz_DESgc__0=nz_DESgc__0,
                        nz_DESgc__1=nz_DESgc__1,
                        nz_DESgc__2=nz_DESgc__2,
                        nz_DESgc__3=nz_DESgc__3,
                        nz_DESgc__4=nz_DESgc__4);

data = meta.data
cov = meta.cov

Γ = sqrt(cov)
iΓ = inv(Γ)
data = iΓ * data

init_params=[0.30, 0.05, 0.67, 0.81, 0.95]
init_params = [init_params; 
    nz_k0;
    nz_k1;
    nz_k2;
    nz_k3;
    nz_k4]

@model function model(data;
    meta=meta, 
    files=files)

    #KiDS priors
    Ωm ~ Uniform(0.2, 0.6)
    Ωb ~ Uniform(0.028, 0.065)
    h ~ Truncated(Normal(0.72, 0.05), 0.64, 0.82)
    σ8 ~ Uniform(0.4, 1.2)
    ns ~ Uniform(0.84, 1.1)

    cosmology = Cosmology(Ωm=Ωm,  Ωb=Ωb, h=h, ns=ns, σ8=σ8,
        tk_mode=:EisHu,
        pk_mode=:Halofit)

    n = length(nz_k0)
    DESgc__0_nz = zeros(cosmology.settings.cosmo_type, n)
    DESgc__1_nz = zeros(cosmology.settings.cosmo_type, n)
    DESgc__2_nz = zeros(cosmology.settings.cosmo_type, n)
    DESgc__3_nz = zeros(cosmology.settings.cosmo_type, n)
    DESgc__4_nz = zeros(cosmology.settings.cosmo_type, n)
    for i in 1:n
        DESgc__0_nz[i] ~ TruncatedNormal(nz_k0[i], sqrt.(diag(cov_k0))[i], -0.07, 0.5)
        DESgc__1_nz[i] ~ TruncatedNormal(nz_k1[i], sqrt.(diag(cov_k1))[i], -0.07, 0.5)
        DESgc__2_nz[i] ~ TruncatedNormal(nz_k2[i], sqrt.(diag(cov_k2))[i], -0.07, 0.5)
        DESgc__3_nz[i] ~ TruncatedNormal(nz_k3[i], sqrt.(diag(cov_k3))[i], -0.07, 0.5)
        DESgc__4_nz[i] ~ TruncatedNormal(nz_k4[i], sqrt.(diag(cov_k4))[i], -0.07, 0.5)
    end

    nuisances = Dict("DESgc__0_b" => 1.484,
                     "DESgc__1_b" => 1.805,
                     "DESgc__2_b" => 1.776,
                     "DESgc__3_b" => 2.168,
                     "DESgc__4_b" => 2.23,
                     "DESgc__0_nz" => DESgc__0_nz,
                     "DESgc__1_nz" => DESgc__1_nz,
                     "DESgc__2_nz" => DESgc__2_nz,
                     "DESgc__3_nz" => DESgc__3_nz,
                     "DESgc__4_nz" => DESgc__4_nz,
                     "DESwl__0_m" => 0.018,
                     "DESwl__1_m" => 0.014,
                     "DESwl__2_m" => 0.01,
                     "DESwl__3_m" => 0.004,
                     "A_IA" => 0.294,
                     "alpha_IA" => 0.378)

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
folname = string("DES_gcgc_nz_num_old_TAP_", TAP,  "_init_ϵ_", init_ϵ)
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
