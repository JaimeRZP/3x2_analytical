ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
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


method = "sompz"
sacc_path = "../../data/CosmoDC2/summary_statistics_fourier_tjpcov.sacc"
yaml_path = "../../data/CosmoDC2/gcgc_gcwl_wlwl.yml"
nz_path = string("../../data/CosmoDC2/nzs_", method, "/PCA_priors/")

sacc_file = sacc.Sacc().load_fits(sacc_path)
yaml_file = YAML.load_file(yaml_path)

nz_lens_0 = npzread(string(nz_path, "PCA_lens_0.npz"))
nz_lens_1 = npzread(string(nz_path, "PCA_lens_1.npz"))
nz_lens_2 = npzread(string(nz_path, "PCA_lens_2.npz"))
nz_lens_3 = npzread(string(nz_path, "PCA_lens_3.npz"))
nz_lens_4 = npzread(string(nz_path, "PCA_lens_4.npz"))
nz_source_0 = npzread(string(nz_path, "PCA_source_0.npz"))
nz_source_1 = npzread(string(nz_path, "PCA_source_1.npz"))
nz_source_2 = npzread(string(nz_path, "PCA_source_2.npz"))
nz_source_3 = npzread(string(nz_path, "PCA_source_3.npz"))
nz_source_4 = npzread(string(nz_path, "PCA_source_4.npz"))

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

W_source_0 = nz_source_0["W"]
W_source_1 = nz_source_1["W"]
W_source_2 = nz_source_2["W"]
W_source_3 = nz_source_3["W"]
W_source_4 = nz_source_4["W"]
W_lens_0 = nz_lens_0["W"]
W_lens_1 = nz_lens_1["W"]
W_lens_2 = nz_lens_2["W"]
W_lens_3 = nz_lens_3["W"]
W_lens_4 = nz_lens_4["W"]

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

cov = 0.03*meta.cov
Γ = sqrt(cov)
iΓ = inv(Γ)
data = iΓ * meta.data

init_alphas = zeros(50)
init_params=[0.27, 0.42, 0.7, 0.77, 0.99]
init_params = [init_params; init_alphas;
                [1.0, 1.0, 1.0, 1.0, 1.0,
                0.0]]

nz_source_0 = zeros(Real, 100)
nz_source_1 = zeros(Real, 100)
nz_source_2 = zeros(Real, 100)
nz_source_3 = zeros(Real, 100)
nz_source_4 = zeros(Real, 100)
nz_lens_0 = zeros(Real, 100)
nz_lens_1 = zeros(Real, 100)
nz_lens_2 = zeros(Real, 100)
nz_lens_3 = zeros(Real, 100)
nz_lens_4 = zeros(Real, 100)

function make_theory(;
    Ωm=0.27347, σ8=0.779007, Ωb=0.04217, h=0.71899, ns=0.99651,
    lens_0_b=0.879118, 
    lens_1_b=1.05894, 
    lens_2_b=1.22145, 
    lens_3_b=1.35065, 
    lens_4_b=1.58909,
    alphas_source_0=zeros(5), 
    alphas_source_1=zeros(5), 
    alphas_source_2=zeros(5), 
    alphas_source_3=zeros(5),
    alphas_source_4=zeros(5),
    alphas_lens_0=zeros(5),
    alphas_lens_1=zeros(5),
    alphas_lens_2=zeros(5),
    alphas_lens_3=zeros(5),
    alphas_lens_4=zeros(5),
    A_IA=0.25179439,
    meta=meta, files=files)

    nz_lens_0 .= nz_k0 + W_lens_0 * alphas_lens_0
    nz_lens_1 .= nz_k1 + W_lens_1 * alphas_lens_1
    nz_lens_2 .= nz_k2 + W_lens_2 * alphas_lens_2
    nz_lens_3 .= nz_k3 + W_lens_3 * alphas_lens_3
    nz_lens_4 .= nz_k4 + W_lens_4 * alphas_lens_4
    nz_source_0 .= nz_k5 + W_source_0 * alphas_source_0
    nz_source_1 .= nz_k6 + W_source_1 * alphas_source_1
    nz_source_2 .= nz_k7 + W_source_2 * alphas_source_2
    nz_source_3 .= nz_k8 + W_source_3 * alphas_source_3
    nz_source_4 .= nz_k9 + W_source_4 * alphas_source_4

    nuisances = Dict(
        "lens_0_b"    => lens_0_b,
        "lens_1_b"    => lens_1_b,
        "lens_2_b"    => lens_2_b,
        "lens_3_b"    => lens_3_b,
        "lens_4_b"    => lens_4_b,
        "lens_0_nz"   => nz_lens_0,
        "lens_1_nz"   => nz_lens_1,
        "lens_2_nz"   => nz_lens_2,
        "lens_3_nz"   => nz_lens_3,
        "lens_4_nz"   => nz_lens_4,
        "source_0_nz" => nz_source_0,
        "source_1_nz" => nz_source_1,
        "source_2_nz" => nz_source_2,
        "source_3_nz" => nz_source_3,
        "source_4_nz" => nz_source_4,
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

    alphas_lens_0 ~ filldist(truncated(Normal(0, 1), -3, 3), 5)
    alphas_lens_1 ~ filldist(truncated(Normal(0, 1), -3, 3), 5)
    alphas_lens_2 ~ filldist(truncated(Normal(0, 1), -3, 3), 5)
    alphas_lens_3 ~ filldist(truncated(Normal(0, 1), -3, 3), 5)
    alphas_lens_4 ~ filldist(truncated(Normal(0, 1), -3, 3), 5)
    alphas_source_0 ~ filldist(truncated(Normal(0, 1), -3, 3), 5)
    alphas_source_1 ~ filldist(truncated(Normal(0, 1), -3, 3), 5)
    alphas_source_2 ~ filldist(truncated(Normal(0, 1), -3, 3), 5)
    alphas_source_3 ~ filldist(truncated(Normal(0, 1), -3, 3), 5)
    alphas_source_4 ~ filldist(truncated(Normal(0, 1), -3, 3), 5)
    lens_0_b ~ Uniform(0.5, 2.5)
    lens_1_b ~ Uniform(0.5, 2.5)
    lens_2_b ~ Uniform(0.5, 2.5)
    lens_3_b ~ Uniform(0.5, 2.5)
    lens_4_b ~ Uniform(0.5, 2.5)
    A_IA ~ Uniform(-1.0, 1.0)

    theory := make_theory(Ωm=Ωm, Ωb=Ωb, h=h, σ8=σ8, ns=ns,
                          alphas_source_0=alphas_source_0,
                          alphas_source_1=alphas_source_1,
                          alphas_source_2=alphas_source_2,
                          alphas_source_3=alphas_source_3,
                          alphas_source_4=alphas_source_4,
                          alphas_lens_0=alphas_lens_0,
                          alphas_lens_1=alphas_lens_1,
                          alphas_lens_2=alphas_lens_2,
                          alphas_lens_3=alphas_lens_3,
                          alphas_lens_4=alphas_lens_4,
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

iterations = 200
adaptation = 100
TAP = 0.65
init_ϵ1 = 0.007
init_ϵ2 = 0.05
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
folpath = string("../../", method, "_fake_chains/numerical/")
folname = string("Y1_3x2_Gibbs_PCA_num",
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
        last_n = parse(Int, last_chain[7:end-4])
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
println(string("Done with chain ", last_n+1,"!"))