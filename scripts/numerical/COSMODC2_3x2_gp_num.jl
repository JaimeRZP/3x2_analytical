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
nz_path = string("../../data/CosmoDC2/image_nzs_", method, "_priors/")
gp_path = string("../../data/CosmoDC2/image_gp_", method, "_priors/")

sacc_file = sacc.Sacc().load_fits(sacc_path)
yaml_file = YAML.load_file(yaml_path)

nz_lens_0 = npzread(string(nz_path, "nz_lens_0.npz"))
nz_lens_1 = npzread(string(nz_path, "nz_lens_1.npz"))
nz_lens_2 = npzread(string(nz_path, "nz_lens_2.npz"))
nz_lens_3 = npzread(string(nz_path, "nz_lens_3.npz"))
nz_lens_4 = npzread(string(nz_path, "nz_lens_4.npz"))
nz_source_0 = npzread(string(nz_path, "nz_source_0.npz"))
nz_source_1 = npzread(string(nz_path, "nz_source_1.npz"))
nz_source_2 = npzread(string(nz_path, "nz_source_2.npz"))
nz_source_3 = npzread(string(nz_path, "nz_source_3.npz"))
nz_source_4 = npzread(string(nz_path, "nz_source_4.npz"))

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

zs_k0, nz_k0 = LimberJack.nz_interpolate(zs_k0, nz_k0, 1000; mode="cubic")
zs_k1, nz_k1 = LimberJack.nz_interpolate(zs_k1, nz_k1, 1000; mode="cubic")
zs_k2, nz_k2 = LimberJack.nz_interpolate(zs_k2, nz_k2, 1000; mode="cubic")
zs_k3, nz_k3 = LimberJack.nz_interpolate(zs_k3, nz_k3, 1000; mode="cubic")
zs_k4, nz_k4 = LimberJack.nz_interpolate(zs_k4, nz_k4, 1000; mode="cubic")
zs_k5, nz_k5 = LimberJack.nz_interpolate(zs_k5, nz_k5, 1000; mode="cubic")
zs_k6, nz_k6 = LimberJack.nz_interpolate(zs_k6, nz_k6, 1000; mode="cubic")
zs_k7, nz_k7 = LimberJack.nz_interpolate(zs_k7, nz_k7, 1000; mode="cubic")
zs_k8, nz_k8 = LimberJack.nz_interpolate(zs_k8, nz_k8, 1000; mode="cubic")
zs_k9, nz_k9 = LimberJack.nz_interpolate(zs_k9, nz_k9, 1000; mode="cubic")

nz_lens_0 = Dict("z"=>zs_k0, "dndz"=>nz_k0)
nz_lens_1 = Dict("z"=>zs_k1, "dndz"=>nz_k1)
nz_lens_2 = Dict("z"=>zs_k2, "dndz"=>nz_k2)
nz_lens_3 = Dict("z"=>zs_k3, "dndz"=>nz_k3)
nz_lens_4 = Dict("z"=>zs_k4, "dndz"=>nz_k4)
nz_source_0 = Dict("z"=>zs_k5, "dndz"=>nz_k5)
nz_source_1 = Dict("z"=>zs_k6, "dndz"=>nz_k6)
nz_source_2 = Dict("z"=>zs_k7, "dndz"=>nz_k7)
nz_source_3 = Dict("z"=>zs_k8, "dndz"=>nz_k8)
nz_source_4 = Dict("z"=>zs_k9, "dndz"=>nz_k9)

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

gp_source_0 = npzread(string(gp_path, "gp_source_0.npz"))
gp_source_1 = npzread(string(gp_path, "gp_source_1.npz"))
gp_source_2 = npzread(string(gp_path, "gp_source_2.npz"))
gp_source_3 = npzread(string(gp_path, "gp_source_3.npz"))
gp_source_4 = npzread(string(gp_path, "gp_source_4.npz"))
gp_lens_0 = npzread(string(gp_path, "gp_lens_0.npz"))
gp_lens_1 = npzread(string(gp_path, "gp_lens_1.npz"))
gp_lens_2 = npzread(string(gp_path, "gp_lens_2.npz"))
gp_lens_3 = npzread(string(gp_path, "gp_lens_3.npz"))
gp_lens_4 = npzread(string(gp_path, "gp_lens_4.npz"))

gp_zs_source_0 = gp_source_0["z"]
gp_zs_source_1 = gp_source_1["z"]
gp_zs_source_2 = gp_source_2["z"]
gp_zs_source_3 = gp_source_3["z"]
gp_zs_source_4 = gp_source_4["z"]
gp_zs_lens_0 = gp_lens_0["z"]
gp_zs_lens_1 = gp_lens_1["z"]
gp_zs_lens_2 = gp_lens_2["z"]
gp_zs_lens_3 = gp_lens_3["z"]
gp_zs_lens_4 = gp_lens_4["z"]

gp_mean_source_0 = gp_source_0["mean"]
gp_mean_source_1 = gp_source_1["mean"]
gp_mean_source_2 = gp_source_2["mean"]
gp_mean_source_3 = gp_source_3["mean"]
gp_mean_source_4 = gp_source_4["mean"]
gp_mean_lens_0 = gp_lens_0["mean"]
gp_mean_lens_1 = gp_lens_1["mean"]
gp_mean_lens_2 = gp_lens_2["mean"]
gp_mean_lens_3 = gp_lens_3["mean"]
gp_mean_lens_4 = gp_lens_4["mean"]

gp_chol_source_0 = gp_lens_0["chol"]
gp_chol_source_1 = gp_lens_1["chol"]
gp_chol_source_2 = gp_lens_2["chol"]
gp_chol_source_3 = gp_lens_3["chol"]
gp_chol_source_4 = gp_lens_4["chol"]
gp_chol_lens_0 = gp_lens_0["chol"]
gp_chol_lens_1 = gp_lens_1["chol"]
gp_chol_lens_2 = gp_lens_2["chol"]
gp_chol_lens_3 = gp_lens_3["chol"]
gp_chol_lens_4 = gp_lens_4["chol"]

gp_W_source_0 = gp_source_0["W"]
gp_W_source_1 = gp_source_1["W"]
gp_W_source_2 = gp_source_2["W"]
gp_W_source_3 = gp_source_3["W"]
gp_W_source_4 = gp_source_4["W"]
gp_W_lens_0 = gp_lens_0["W"]
gp_W_lens_1 = gp_lens_1["W"]
gp_W_lens_2 = gp_lens_2["W"]
gp_W_lens_3 = gp_lens_3["W"]
gp_W_lens_4 = gp_lens_4["W"]

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

init_alphas = zeros(10)
init_params=[0.30, 0.5, 0.67, 0.81, 0.95]
init_params = [init_params; init_alphas;
                [1.0, 1.0, 1.0, 1.0, 1.0,
                0.0]]

nz_source_0 = zeros(Real, 1000)
nz_source_1 = zeros(Real, 1000)
nz_source_2 = zeros(Real, 1000)
nz_source_3 = zeros(Real, 1000)
nz_source_4 = zeros(Real, 1000)
nz_lens_0 = zeros(Real, 1000)
nz_lens_1 = zeros(Real, 1000)
nz_lens_2 = zeros(Real, 1000)
nz_lens_3 = zeros(Real, 1000)
nz_lens_4 = zeros(Real, 1000)

function nz_itp(q, nq, z)
    dq = mean(q[2:end] - q[1:end-1])
    q_range = q[1]:dq:q[end]
    nz_int = cubic_spline_interpolation(q_range, nq;
        extrapolation_bc=Line())
    return nz_int(z)
end

function make_theory(;
    Ωm=0.27347, σ8=0.779007, Ωb=0.04217, h=0.71899, ns=0.99651,
    lens_1_b=0.879118, lens_2_b=1.05894, lens_3_b=1.22145, lens_4_b=1.35065, lens_5_b=1.58909,
    alphas_source_0=zeros(10), 
    alphas_source_1=zeros(10), 
    alphas_source_2=zeros(10), 
    alphas_source_3=zeros(10),
    alphas_source_4=zeros(10),
    alphas_lens_0=zeros(10),
    alphas_lens_1=zeros(10),
    alphas_lens_2=zeros(10),
    alphas_lens_3=zeros(10),
    alphas_lens_4=zeros(10),
    A_IA=0.25179439,
    meta=meta, files=files)

    nz_source_0 .= gp_W_source_0 * gp_chol_source_0 * alphas_source_0
    nz_source_0 .+= gp_mean_source_0
    nz_source_1 .= gp_W_source_1 * gp_chol_source_1 * alphas_source_1
    nz_source_1 .+= gp_mean_source_1
    nz_source_2 .= gp_W_source_2 * gp_chol_source_2 * alphas_source_2
    nz_source_2 .+= gp_mean_source_2
    nz_source_3 .= gp_W_source_3 * gp_chol_source_3 * alphas_source_3
    nz_source_3 .+= gp_mean_source_3
    nz_source_4 .= gp_W_source_4 * gp_chol_source_4 * alphas_source_4
    nz_source_4 .+= gp_mean_source_4
    nz_lens_0 .= gp_W_lens_0 * gp_chol_lens_0 * alphas_lens_0
    nz_lens_0 .+= gp_mean_lens_0
    nz_lens_1 .= gp_W_lens_1 * gp_chol_lens_1 * alphas_lens_1
    nz_lens_1 .+= gp_mean_lens_1
    nz_lens_2 .= gp_W_lens_2 * gp_chol_lens_2 * alphas_lens_2
    nz_lens_2 .+= gp_mean_lens_2
    nz_lens_3 .= gp_W_lens_3 * gp_chol_lens_3 * alphas_lens_3
    nz_lens_3 .+= gp_mean_lens_3
    nz_lens_4 .= gp_W_lens_4 * gp_chol_lens_4 * alphas_lens_4
    nz_lens_4 .+= gp_mean_lens_4

    nuisances = Dict(
        "lens_1_b"    => lens_1_b,
        "lens_2_b"    => lens_2_b,
        "lens_3_b"    => lens_3_b,
        "lens_4_b"    => lens_4_b,
        "lens_5_b"    => lens_5_b,
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

fake_data = make_theory();
println("Theory created!")
fake_data = iΓ * fake_data
data = fake_data

@model function model(data)
    Ωm ~ Uniform(0.2, 0.6)
    Ωbb ~ Uniform(0.28, 0.65) # 10*Ωb 
    Ωb := 0.1*Ωbb 
    h ~ Truncated(Normal(0.72, 0.05), 0.64, 0.82)
    σ8 ~ Uniform(0.4, 1.2)
    ns ~ Uniform(0.84, 1.1)

    alphas_lens_0 ~ filldist(truncated(Normal(0, 1), -3, 3), 10)
    alphas_lens_1 ~ filldist(truncated(Normal(0, 1), -3, 3), 10)
    alphas_lens_2 ~ filldist(truncated(Normal(0, 1), -3, 3), 10)
    alphas_lens_3 ~ filldist(truncated(Normal(0, 1), -3, 3), 10)
    alphas_lens_4 ~ filldist(truncated(Normal(0, 1), -3, 3), 10)
    alphas_source_0 ~ filldist(truncated(Normal(0, 1), -3, 3), 10)
    alphas_source_1 ~ filldist(truncated(Normal(0, 1), -3, 3), 10)
    alphas_source_2 ~ filldist(truncated(Normal(0, 1), -3, 3), 10)
    alphas_source_3 ~ filldist(truncated(Normal(0, 1), -3, 3), 10)
    alphas_source_4 ~ filldist(truncated(Normal(0, 1), -3, 3), 10)
    lens_1_b ~ Uniform(0.5, 2.5)
    lens_2_b ~ Uniform(0.5, 2.5)
    lens_3_b ~ Uniform(0.5, 2.5)
    lens_4_b ~ Uniform(0.5, 2.5)
    lens_5_b ~ Uniform(0.5, 2.5)
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
                          lens_1_b=lens_1_b, lens_2_b=lens_2_b,
                          lens_3_b=lens_3_b, lens_4_b=lens_4_b,
                          lens_5_b=lens_5_b, A_IA=A_IA)
    ttheory = iΓ * theory
    d = data - ttheory
    Xi2 := dot(d, d)
    data ~ MvNormal(ttheory, I)
end

iterations = 200
adaptation = 100
TAP = 0.65
init_ϵ1 = 0.03
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
folpath = "../../nuisance_fake_chains/numerical/"
folname = string("CosmoDC2_3x2_Gibbs_indep_gp_fixed_num",
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
    :lens_1_b, :lens_2_b, :lens_3_b, :lens_4_b, :lens_5_b;
    init_ϵ=init_ϵ1, max_depth=max_depth),
    NUTS(adaptation, TAP,
    :A_IA, 
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
