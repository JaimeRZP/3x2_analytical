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
using Interpolations
sacc = pyimport("sacc");


method = "bpz"
sacc_path = "../../data/CosmoDC2/summary_statistics_fourier_tjpcov.sacc"
yaml_path = "../../data/CosmoDC2/gcgc_gcwl_wlwl.yml"
nz_path = string("../../data/CosmoDC2/image_wzdz_", method, "_priors/")

sacc_file = sacc.Sacc().load_fits(sacc_path)
yaml_file = YAML.load_file(yaml_path)

nz_lens_0 = npzread(string(nz_path, "wzdz_lens_0.npz"))
nz_lens_1 = npzread(string(nz_path, "wzdz_lens_1.npz"))
nz_lens_2 = npzread(string(nz_path, "wzdz_lens_2.npz"))
nz_lens_3 = npzread(string(nz_path, "wzdz_lens_3.npz"))
nz_lens_4 = npzread(string(nz_path, "wzdz_lens_4.npz"))
nz_source_0 = npzread(string(nz_path, "wzdz_source_0.npz"))
nz_source_1 = npzread(string(nz_path, "wzdz_source_1.npz"))
nz_source_2 = npzread(string(nz_path, "wzdz_source_2.npz"))
nz_source_3 = npzread(string(nz_path, "wzdz_source_3.npz"))
nz_source_4 = npzread(string(nz_path, "wzdz_source_4.npz"))

zs_k0, nz_k0 = nz_lens_0["z"], nz_lens_0["dndz"]
zs_k1, nz_k1 = nz_lens_1["z"], nz_lens_1["dndz"]
zs_k2, nz_k2 = nz_lens_2["z"], nz_lens_2["dndz"]
zs_k3, nz_k3 = nz_lens_3["z"], nz_lens_3["dndz"]
zs_k4, nz_k4 = nz_lens_4["z"], nz_lens_4["dndz"]
zs_k5, nz_k5 = nz_source_0["z"], nz_source_0["dndz"]
zs_k6, nz_k6 = nz_source_1["z"], nz_source_1["dndz"]
zs_k7, nz_k7 = nz_source_2["z"], nz_source_2["dndz"]
zs_k8, nz_k8 = nz_source_3["z"], nz_source_3["dndz"]
zs_k9, nz_k9 = nz_source_4["z"], nz_source_4["dndz"]

mu_k0 = sum(zs_k0 .* nz_k0) / sum(nz_k0)
mu_k1 = sum(zs_k1 .* nz_k1) / sum(nz_k1)
mu_k2 = sum(zs_k2 .* nz_k2) / sum(nz_k2)
mu_k3 = sum(zs_k3 .* nz_k3) / sum(nz_k3)
mu_k4 = sum(zs_k4 .* nz_k4) / sum(nz_k4)
mu_k5 = sum(zs_k5 .* nz_k5) / sum(nz_k5)
mu_k6 = sum(zs_k6 .* nz_k6) / sum(nz_k6)
mu_k7 = sum(zs_k7 .* nz_k7) / sum(nz_k7)
mu_k8 = sum(zs_k8 .* nz_k8) / sum(nz_k8)
mu_k9 = sum(zs_k9 .* nz_k9) / sum(nz_k9)

chol_source_0 = nz_source_0["chol"]
chol_source_1 = nz_source_1["chol"]
chol_source_2 = nz_source_2["chol"]
chol_source_3 = nz_source_3["chol"]
chol_source_4 = nz_source_4["chol"]
chol_lens_0 = nz_lens_0["chol"]
chol_lens_1 = nz_lens_1["chol"]
chol_lens_2 = nz_lens_2["chol"]
chol_lens_3 = nz_lens_3["chol"]
chol_lens_4 = nz_lens_4["chol"]

meta, files = make_data(sacc_file, yaml_file;
                        nz_lens_0=nz_lens_0,
                        nz_lens_1=nz_lens_1,
                        nz_lens_2=nz_lens_2,
                        nz_lens_3=nz_lens_3,
                        nz_lens_4=nz_lens_4,
                        nz_source_0=nz_source_0,
                        nz_source_1=nz_source_1,
                        nz_source_2=nz_source_2,
                        nz_source_3=nz_source_3,
                        nz_source_4=nz_source_4)

meta.types = [ 
    "galaxy_density",
    "galaxy_density",
    "galaxy_density",
    "galaxy_density",
    "galaxy_density",
    "galaxy_shear", 
    "galaxy_shear", 
    "galaxy_shear",
    "galaxy_shear",
    "galaxy_shear"]

cov = meta.cov
Γ = sqrt(cov)
iΓ = inv(Γ)
data = iΓ * meta.data

init_alphas = zeros(20)
init_params=[0.30, 0.5, 0.67, 0.81, 0.95]
init_params = [init_params; init_alphas;
                [1.0, 1.0, 1.0, 1.0, 1.0,
                0.0]]

#lens_0_zs = zeros(Real, 100)
#lens_1_zs = zeros(Real, 100)
#lens_2_zs = zeros(Real, 100)
#lens_3_zs = zeros(Real, 100)
#lens_4_zs = zeros(Real, 100)
#source_0_zs = zeros(Real, 100)
#source_1_zs = zeros(Real, 100)
#source_2_zs = zeros(Real, 100)
#source_3_zs = zeros(Real, 100)
#source_4_zs = zeros(Real, 100)

function make_theory(;
    Ωm=0.27347, σ8=0.779007, Ωb=0.04217, h=0.71899, ns=0.99651,
    lens_0_b=0.879118, 
    lens_1_b=1.05894, 
    lens_2_b=1.22145, 
    lens_3_b=1.35065, 
    lens_4_b=1.58909,
    dz_lens_0=0.0, wz_lens_0=1.0,
    dz_lens_1=0.0, wz_lens_1=1.0,
    dz_lens_2=0.0, wz_lens_2=1.0,
    dz_lens_3=0.0, wz_lens_3=1.0,
    dz_lens_4=0.0, wz_lens_4=1.0,
    dz_source_0=0.0, wz_source_0=1.0,
    dz_source_1=0.0, wz_source_1=1.0,
    dz_source_2=0.0, wz_source_2=1.0,
    dz_source_3=0.0, wz_source_3=1.0,
    dz_source_4=0.0, wz_source_4=1.0,
    A_IA=0.25179439,
    meta=meta, files=files)

    lens_0_zs   = @.((zs_k0 - mu_k0 + dz_lens_0) / wz_lens_0 + mu_k0)
    lens_1_zs   = @.((zs_k1 - mu_k1 + dz_lens_1) / wz_lens_1 + mu_k1)
    lens_2_zs   = @.((zs_k2 - mu_k2 + dz_lens_2) / wz_lens_2 + mu_k2)
    lens_3_zs   = @.((zs_k3 - mu_k3 + dz_lens_3) / wz_lens_3 + mu_k3)
    lens_4_zs   = @.((zs_k4 - mu_k4 + dz_lens_4) / wz_lens_4 + mu_k4)
    source_0_zs = @.((zs_k5 - mu_k5 + dz_source_0) / wz_source_0 + mu_k5)
    source_1_zs = @.((zs_k6 - mu_k6 + dz_source_1) / wz_source_1 + mu_k6)
    source_2_zs = @.((zs_k7 - mu_k7 + dz_source_2) / wz_source_2 + mu_k7)
    source_3_zs = @.((zs_k8 - mu_k8 + dz_source_3) / wz_source_3 + mu_k8)
    source_4_zs = @.((zs_k9 - mu_k9 + dz_source_4) / wz_source_4 + mu_k9)

    nuisances = Dict(
    "lens_0_b"    => lens_0_b,
    "lens_1_b"    => lens_1_b,
    "lens_2_b"    => lens_2_b,
    "lens_3_b"    => lens_3_b,
    "lens_4_b"    => lens_4_b,
    "lens_0_zs"   => lens_0_zs,
    "lens_1_zs"   => lens_1_zs,
    "lens_2_zs"   => lens_2_zs,
    "lens_3_zs"   => lens_3_zs,
    "lens_4_zs"   => lens_4_zs,
    "source_0_zs" => source_0_zs,
    "source_1_zs" => source_1_zs,
    "source_2_zs" => source_2_zs,
    "source_3_zs" => source_3_zs,
    "source_4_zs" => source_4_zs,
    "A_IA"        => A_IA)
       
    cosmology = Cosmology(Ωm=Ωm, Ωb=Ωb, h=h, ns=ns, σ8=σ8,
        tk_mode=:EisHu,
        pk_mode=:Halofit)

    return Theory(cosmology, meta, files; 
             Nuisances=nuisances)
end

@model function model(data)
    Ωm ~ Uniform(0.2, 0.6)
    Ωbb ~ Uniform(0.28, 0.65) # 10*Ωb 
    Ωb := 0.1*Ωbb 
    h ~ Truncated(Normal(0.72, 0.05), 0.64, 0.82)
    σ8 ~ Uniform(0.4, 1.2)
    ns ~ Uniform(0.84, 1.1)

    alphas_lens_0 ~ filldist(truncated(Normal(0, 1), -3, 3), 2)
    alphas_lens_1 ~ filldist(truncated(Normal(0, 1), -3, 3), 2)
    alphas_lens_2 ~ filldist(truncated(Normal(0, 1), -3, 3), 2)
    alphas_lens_3 ~ filldist(truncated(Normal(0, 1), -3, 3), 2)
    alphas_lens_4 ~ filldist(truncated(Normal(0, 1), -3, 3), 2)
    alphas_source_0 ~ filldist(truncated(Normal(0, 1), -3, 3), 2)
    alphas_source_1 ~ filldist(truncated(Normal(0, 1), -3, 3), 2)
    alphas_source_2 ~ filldist(truncated(Normal(0, 1), -3, 3), 2)
    alphas_source_3 ~ filldist(truncated(Normal(0, 1), -3, 3), 2)
    alphas_source_4 ~ filldist(truncated(Normal(0, 1), -3, 3), 2)
    lens_0_b ~ Uniform(0.5, 2.5)
    lens_1_b ~ Uniform(0.5, 2.5)
    lens_2_b ~ Uniform(0.5, 2.5)
    lens_3_b ~ Uniform(0.5, 2.5)
    lens_4_b ~ Uniform(0.5, 2.5)
    A_IA ~ Uniform(-1.0, 1.0)

    snw_lens_0 = chol_lens_0 * alphas_lens_0
    dz_lens_0 := snw_lens_0[1]
    wz_lens_0 := 1 + snw_lens_0[2]

    snw_lens_1 = chol_lens_1 * alphas_lens_1
    dz_lens_1 := snw_lens_1[1]
    wz_lens_1 := 1 + snw_lens_1[2]

    snw_lens_2 = chol_lens_2 * alphas_lens_2
    dz_lens_2 := snw_lens_2[1]
    wz_lens_2 := 1 + snw_lens_2[2]

    snw_lens_3 = chol_lens_3 * alphas_lens_3
    dz_lens_3 := snw_lens_3[1]
    wz_lens_3 := 1 + snw_lens_3[2]

    snw_lens_4 = chol_lens_4 * alphas_lens_4
    dz_lens_4 := snw_lens_4[1]
    wz_lens_4 := 1 + snw_lens_4[2]

    snw_source_0 = chol_source_0 * alphas_source_0
    dz_source_0 := snw_source_0[1]
    wz_source_0 := 1 + snw_source_0[2]

    snw_source_1 = chol_source_1 * alphas_source_1
    dz_source_1 := snw_source_1[1]
    wz_source_1 := 1 + snw_source_1[2]

    snw_source_2 = chol_source_2 * alphas_source_2
    dz_source_2 := snw_source_2[1]
    wz_source_2 := 1 + snw_source_2[2]

    snw_source_3 = chol_source_3 * alphas_source_3
    dz_source_3 := snw_source_3[1]
    wz_source_3 := 1 + snw_source_3[2]

    snw_source_4 = chol_source_4 * alphas_source_4
    dz_source_4 := snw_source_4[1]
    wz_source_4 := 1 + snw_source_4[2]

    theory := make_theory(Ωm=Ωm, Ωb=Ωb, h=h, σ8=σ8, ns=ns,
                        dz_lens_0=dz_lens_0, wz_lens_0=wz_lens_0,
                        dz_lens_1=dz_lens_1, wz_lens_1=wz_lens_1,
                        dz_lens_2=dz_lens_2, wz_lens_2=wz_lens_2,
                        dz_lens_3=dz_lens_3, wz_lens_3=wz_lens_3,
                        dz_lens_4=dz_lens_4, wz_lens_4=wz_lens_4,
                        dz_source_0=dz_source_0, wz_source_0=wz_source_0,
                        dz_source_1=dz_source_1, wz_source_1=wz_source_1,
                        dz_source_2=dz_source_2, wz_source_2=wz_source_2,
                        dz_source_3=dz_source_3, wz_source_3=wz_source_3,
                        dz_source_4=dz_source_4, wz_source_4=wz_source_4,
                        lens_0_b=lens_0_b, 
                        lens_1_b=lens_1_b,
                        lens_2_b=lens_2_b, 
                        lens_3_b=lens_3_b,
                        lens_4_b=lens_4_b, 
                        A_IA=A_IA)
    ttheory = iΓ * theory
    d = data - ttheory
    Xi2 := dot(d, d)
    data ~ MvNormal(ttheory, I)
end

iterations = 300
adaptation = 100
TAP = 0.65
init_ϵ1 = 0.01
init_ϵ2 = 0.03
max_depth = 8

println("sampling settings: ")
println("iterations ", iterations)
println("TAP ", TAP)
println("init_ϵ1 ", init_ϵ1)
println("init_ϵ2 ", init_ϵ2)
println("max_depth ", max_depth)
println("adaptation ", adaptation)
#println("nchains ", nchains)

# Start sampling.
folpath = "../../real_chains/numerical/"
folname = string("CosmoDC2_3x2_Gibbs_wzdz_num",
    "_TAP_", TAP,
    "_init_ϵ1_", init_ϵ1, 
    "_init_ϵ2_", init_ϵ2,
)
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
#sampler = NUTS(adaptation, TAP;
#    init_ϵ=init_ϵ, max_depth=max_depth)
sampler = Gibbs(
    NUTS(adaptation, TAP,
    :Ωm, :Ωbb, :h, :σ8, :ns,
    :lens_0_b,
    :lens_1_b,
    :lens_2_b,
    :lens_3_b,
    :lens_4_b,
    :A_IA;
    init_ϵ=init_ϵ1, max_depth=max_depth),
    NUTS(adaptation, TAP,
    :alphas_lens_0,
    :alphas_lens_1,
    :alphas_lens_2,
    :alphas_lens_3,
    :alphas_lens_4,
    :alphas_source_0,
    :alphas_source_1,
    :alphas_source_2,
    :alphas_source_3,
    :alphas_source_4;
    init_ϵ=init_ϵ2, max_depth=max_depth))
chain = sample(cond_model, sampler, iterations;
                init_params=init_params,
                progress=true, save_state=true)

# Save the actual chain.       
@save joinpath(folname, string("chain_", last_n+1,".jls")) chain
CSV.write(joinpath(folname, string("chain_", last_n+1,".csv")), chain)
CSV.write(joinpath(folname, string("summary_", last_n+1,".csv")), describe(chain)[1])
npzwrite(joinpath(folname, string("data_", last_n+1,".npz")), data=make_theory())
