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
using OptimizationOptimJL: NelderMead
sacc = pyimport("sacc");


method = "sompz"
sacc_path = "../../data/CosmoDC2/summary_statistics_fourier_tjpcov.sacc"
yaml_path = "../../data/CosmoDC2/gcgc_gcwl_wlwl.yml"
nz_path = string("../../data/CosmoDC2/nzs_", method, "/")
nz_priors =string("../../data/CosmoDC2/nzs_", method, "/PCA_5_priors/")
param_path = string("../../data/CosmoDC2/nzs_", method, "/PCA_5_priors/pca_params.npz")

sacc_file = sacc.Sacc().load_fits(sacc_path)
yaml_file = YAML.load_file(yaml_path)
PCA_params = npzread(param_path)

nz_lens_0 =  npzread(string(nz_path, "lens_0.npz"))
nz_lens_1 =  npzread(string(nz_path, "lens_1.npz"))
nz_lens_2 =  npzread(string(nz_path, "lens_2.npz"))
nz_lens_3 =  npzread(string(nz_path, "lens_3.npz"))
nz_lens_4 =  npzread(string(nz_path, "lens_4.npz"))
nz_source_0 =  npzread(string(nz_path, "source_0.npz"))
nz_source_1 =  npzread(string(nz_path, "source_1.npz"))
nz_source_2 =  npzread(string(nz_path, "source_2.npz"))
nz_source_3 =  npzread(string(nz_path, "source_3.npz"))
nz_source_4 =  npzread(string(nz_path, "source_4.npz"))

W_lens_0 = npzread(string(nz_priors, "PCA_lens_0.npz"))["W"]
W_lens_1 = npzread(string(nz_priors, "PCA_lens_1.npz"))["W"]
W_lens_2 = npzread(string(nz_priors, "PCA_lens_2.npz"))["W"]
W_lens_3 = npzread(string(nz_priors, "PCA_lens_3.npz"))["W"]
W_lens_4 = npzread(string(nz_priors, "PCA_lens_4.npz"))["W"]
W_source_0 = npzread(string(nz_priors, "PCA_source_0.npz"))["W"]
W_source_1 = npzread(string(nz_priors, "PCA_source_1.npz"))["W"]
W_source_2 = npzread(string(nz_priors, "PCA_source_2.npz"))["W"]
W_source_3 = npzread(string(nz_priors, "PCA_source_3.npz"))["W"]
W_source_4 = npzread(string(nz_priors, "PCA_source_4.npz"))["W"]

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

scale = 0.03
cov = scale*meta.cov
Γ = sqrt(cov)
iΓ = inv(Γ)

function make_theory(;
    Ωm=0.27347, Ωb=0.04217, h=0.71899, σ8=0.779007, ns=0.99651,
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

    nz_lens_0 = files["nz_lens_0"][2]
    nz_lens_1 = files["nz_lens_1"][2]
    nz_lens_2 = files["nz_lens_2"][2]
    nz_lens_3 = files["nz_lens_3"][2]
    nz_lens_4 = files["nz_lens_4"][2]
    nz_source_0 = files["nz_source_0"][2]
    nz_source_1 = files["nz_source_1"][2]
    nz_source_2 = files["nz_source_2"][2]
    nz_source_3 = files["nz_source_3"][2]
    nz_source_4 = files["nz_source_4"][2]

    nuisances = Dict(
        "lens_0_b"    => lens_0_b,
        "lens_1_b"    => lens_1_b,
        "lens_2_b"    => lens_2_b,
        "lens_3_b"    => lens_3_b,
        "lens_4_b"    => lens_4_b,
        "lens_0_nz"   => nz_lens_0 + W_lens_0 * alphas_lens_0,
        "lens_1_nz"   => nz_lens_1 + W_lens_1 * alphas_lens_1,
        "lens_2_nz"   => nz_lens_2 + W_lens_2 * alphas_lens_2,
        "lens_3_nz"   => nz_lens_3 + W_lens_3 * alphas_lens_3,
        "lens_4_nz"   => nz_lens_4 + W_lens_4 * alphas_lens_4,
        "source_0_nz" => nz_source_0 + W_source_0 * alphas_source_0,
        "source_1_nz" => nz_source_1 + W_source_1 * alphas_source_1,
        "source_2_nz" => nz_source_2 + W_source_2 * alphas_source_2,
        "source_3_nz" => nz_source_3 + W_source_3 * alphas_source_3,
        "source_4_nz" => nz_source_4 + W_source_4 * alphas_source_4,
        "A_IA"        => A_IA)
    
    cosmology = Cosmology(Ωm=Ωm, Ωb=Ωb, h=h, ns=ns, σ8=σ8,
        tk_mode=:EisHu,
        pk_mode=:Halofit)

    return Theory(cosmology, meta, files; 
            Nuisances=nuisances)
end

@model function model(data;
    alphas_lens_0=zeros(5),
    alphas_lens_1=zeros(5),
    alphas_lens_2=zeros(5),
    alphas_lens_3=zeros(5),
    alphas_lens_4=zeros(5),
    alphas_source_0=zeros(5),
    alphas_source_1=zeros(5),
    alphas_source_2=zeros(5),
    alphas_source_3=zeros(5),
    alphas_source_4=zeros(5),
    files=files, meta=meta)
    Ωm ~ Uniform(0.2, 0.4)
    Ωbb ~ Uniform(0.3, 0.5) # 10*Ωb 
    Ωb = 0.1*Ωbb 
    h ~ Truncated(Normal(0.72, 0.05), 0.64, 0.82)
    σ8 ~ Uniform(0.6, 0.9)
    ns ~ Uniform(0.9, 1.0)

    lens_0_b ~ Uniform(0.75, 0.95)
    lens_1_b ~ Uniform(0.95, 1.15)
    lens_2_b ~ Uniform(1.10, 1.30)
    lens_3_b ~ Uniform(1.20, 1.50)
    lens_4_b ~ Uniform(1.40, 1.80)

    theory = make_theory(Ωm=Ωm, Ωb=Ωb, h=h, σ8=σ8, ns=ns,
        alphas_lens_0=alphas_lens_0,
        alphas_lens_1=alphas_lens_1,
        alphas_lens_2=alphas_lens_2,
        alphas_lens_3=alphas_lens_3,
        alphas_lens_4=alphas_lens_4,
        alphas_source_0=alphas_source_0,
        alphas_source_1=alphas_source_1,
        alphas_source_2=alphas_source_2,
        alphas_source_3=alphas_source_3,
        alphas_source_4=alphas_source_4,
        lens_0_b=lens_0_b,
        lens_1_b=lens_1_b,
        lens_2_b=lens_2_b, 
        lens_3_b=lens_3_b,
        lens_4_b=lens_4_b,
        files=files, meta=meta)
    ttheory = iΓ * theory
    d = data - ttheory
    Xi2 := dot(d, d)
    data ~ MvNormal(ttheory, I)
end

for realization in 1:500
    alphas_0 = PCA_params["lens_0"][:, realization]
    alphas_1 = PCA_params["lens_1"][:, realization]
    alphas_2 = PCA_params["lens_2"][:, realization]
    alphas_3 = PCA_params["lens_3"][:, realization]
    alphas_4 = PCA_params["lens_4"][:, realization]
    alphas_5 = PCA_params["source_0"][:, realization]
    alphas_6 = PCA_params["source_1"][:, realization]
    alphas_7 = PCA_params["source_2"][:, realization]
    alphas_8 = PCA_params["source_3"][:, realization]
    alphas_9 = PCA_params["source_4"][:, realization]

    zs_k0, nz_k0 = nz_lens_0["z"], nz_lens_0["photo_hists"][:, realization]
    zs_k1, nz_k1 = nz_lens_1["z"], nz_lens_1["photo_hists"][:, realization]
    zs_k2, nz_k2 = nz_lens_2["z"], nz_lens_2["photo_hists"][:, realization]
    zs_k3, nz_k3 = nz_lens_3["z"], nz_lens_3["photo_hists"][:, realization]
    zs_k4, nz_k4 = nz_lens_4["z"], nz_lens_4["photo_hists"][:, realization]
    zs_k5, nz_k5 = nz_source_0["z"], nz_source_0["photo_hists"][:, realization]
    zs_k6, nz_k6 = nz_source_1["z"], nz_source_1["photo_hists"][:, realization]
    zs_k7, nz_k7 = nz_source_2["z"], nz_source_2["photo_hists"][:, realization]
    zs_k8, nz_k8 = nz_source_3["z"], nz_source_3["photo_hists"][:, realization]
    zs_k9, nz_k9 = nz_source_4["z"], nz_source_4["photo_hists"][:, realization]

    _nz_lens_0 = Vector([zs_k0, nz_k0])
    _nz_lens_1 = Vector([zs_k1, nz_k1])
    _nz_lens_2 = Vector([zs_k2, nz_k2])
    _nz_lens_3 = Vector([zs_k3, nz_k3])
    _nz_lens_4 = Vector([zs_k4, nz_k4])
    _nz_source_0 = Vector([zs_k5, nz_k5])
    _nz_source_1 = Vector([zs_k6, nz_k6])
    _nz_source_2 = Vector([zs_k7, nz_k7])
    _nz_source_3 = Vector([zs_k8, nz_k8])
    _nz_source_4 = Vector([zs_k9, nz_k9])

    _, new_files = make_data(sacc_file, yaml_file;
        nz_lens_0=_nz_lens_0,
        nz_lens_1=_nz_lens_1,
        nz_lens_2=_nz_lens_2,
        nz_lens_3=_nz_lens_3,
        nz_lens_4=_nz_lens_4,
        nz_source_0=_nz_source_0,
        nz_source_1=_nz_source_1,
        nz_source_2=_nz_source_2,
        nz_source_3=_nz_source_3,
        nz_source_4=_nz_source_4)

    fake_data = make_theory(files=new_files);
    fake_data = iΓ * fake_data
    data = fake_data
    folpath = string("../../", method, "_fake_chains/maximization/Y1_3x2_PCA_maximization/")
    npzwrite(joinpath(folpath, string("data_", realization,".npz")), data=make_theory())
    println(string("Written data for ", realization,"!"))

    # Create the conditioned model with the dz values fixed.
    cond_model = model(data;
        files=files,
        alphas_lens_0=alphas_0,
        alphas_lens_1=alphas_1,
        alphas_lens_2=alphas_2,
        alphas_lens_3=alphas_3,
        alphas_lens_4=alphas_4,
        alphas_source_0=alphas_5,
        alphas_source_1=alphas_6,
        alphas_source_2=alphas_7,
        alphas_source_3=alphas_8,
        alphas_source_4=alphas_9)

    # Find MLE estimate to use as initial parameters.
    mle = maximum_likelihood(cond_model, NelderMead())
    values = mle.values.array
    namess = mle.values.dicts[1].keys
    params = Dict{Symbol, Float64}()
    for (i, name) in enumerate(namess)
        params[Symbol(name)] = values[i]
    end
    if realization == 0
        CSV.write(joinpath(folpath, "samples.csv"), DataFrame(params))
    else
        CSV.write(joinpath(folpath, "samples.csv"), DataFrame(params); append=true)
    end
end