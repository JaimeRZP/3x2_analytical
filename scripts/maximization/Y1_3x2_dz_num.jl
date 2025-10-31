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
param_path = string("../../data/CosmoDC2/nzs_", method, "/dz_priors/dz_params.npz")

sacc_file = sacc.Sacc().load_fits(sacc_path)
yaml_file = YAML.load_file(yaml_path)
dz_params = npzread(string(param_path))

nz_lens_0 = npzread(string(nz_path, "lens_0.npz"))
nz_lens_1 = npzread(string(nz_path, "lens_1.npz"))
nz_lens_2 = npzread(string(nz_path, "lens_2.npz"))
nz_lens_3 = npzread(string(nz_path, "lens_3.npz"))
nz_lens_4 = npzread(string(nz_path, "lens_4.npz"))
nz_source_0 = npzread(string(nz_path, "source_0.npz"))
nz_source_1 = npzread(string(nz_path, "source_1.npz"))
nz_source_2 = npzread(string(nz_path, "source_2.npz"))
nz_source_3 = npzread(string(nz_path, "source_3.npz"))
nz_source_4 = npzread(string(nz_path, "source_4.npz"))

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
    dz_lens_0=0.0,
    dz_lens_1=0.0,
    dz_lens_2=0.0,
    dz_lens_3=0.0,
    dz_lens_4=0.0,
    dz_source_0=0.0,
    dz_source_1=0.0,
    dz_source_2=0.0,
    dz_source_3=0.0,
    dz_source_4=0.0,
    A_IA=0.25179439,
    meta=meta, files=files)

    lens_0_zs   = @.(zs_k0 + dz_lens_0)
    lens_1_zs   = @.(zs_k1 + dz_lens_1)
    lens_2_zs   = @.(zs_k2 + dz_lens_2)
    lens_3_zs   = @.(zs_k3 + dz_lens_3)
    lens_4_zs   = @.(zs_k4 + dz_lens_4)
    source_0_zs = @.(zs_k5 + dz_source_0)
    source_1_zs = @.(zs_k6 + dz_source_1)
    source_2_zs = @.(zs_k7 + dz_source_2)
    source_3_zs = @.(zs_k8 + dz_source_3)
    source_4_zs = @.(zs_k9 + dz_source_4)

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

@model function model(data;
    files=files,
    dz_lens_0=0.0,
    dz_lens_1=0.0,
    dz_lens_2=0.0,
    dz_lens_3=0.0,
    dz_lens_4=0.0,
    dz_source_0=0.0,
    dz_source_1=0.0,
    dz_source_2=0.0,
    dz_source_3=0.0,
    dz_source_4=0.0)

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
        dz_lens_0=dz_lens_0,
        dz_lens_1=dz_lens_1,
        dz_lens_2=dz_lens_2,
        dz_lens_3=dz_lens_3,
        dz_lens_4=dz_lens_4,
        dz_source_0=dz_source_0,
        dz_source_1=dz_source_1,
        dz_source_2=dz_source_2,
        dz_source_3=dz_source_3,
        dz_source_4=dz_source_4,
        lens_0_b=lens_0_b,
        lens_1_b=lens_1_b,
        lens_2_b=lens_2_b, 
        lens_3_b=lens_3_b,
        lens_4_b=lens_4_b, 
        )
    ttheory = iΓ * theory
    d = data - ttheory
    Xi2 := dot(d, d)
    data ~ MvNormal(ttheory, I)
end
    
for realization in 1:500
    dz_0 = dz_params["lens_0"][:, realization]
    dz_1 = dz_params["lens_1"][:, realization]
    dz_2 = dz_params["lens_2"][:, realization]
    dz_3 = dz_params["lens_3"][:, realization]
    dz_4 = dz_params["lens_4"][:, realization]
    dz_5 = dz_params["source_0"][:, realization]
    dz_6 = dz_params["source_1"][:, realization]
    dz_7 = dz_params["source_2"][:, realization]
    dz_8 = dz_params["source_3"][:, realization]
    dz_9 = dz_params["source_4"][:, realization]

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

    files["nz_lens_0"] = Vector([zs_k0, nz_k0])
    files["nz_lens_1"] = Vector([zs_k1, nz_k1])
    files["nz_lens_2"] = Vector([zs_k2, nz_k2])
    files["nz_lens_3"] = Vector([zs_k3, nz_k3])
    files["nz_lens_4"] = Vector([zs_k4, nz_k4])
    files["nz_source_0"] = Vector([zs_k5, nz_k5])
    files["nz_source_1"] = Vector([zs_k6, nz_k6])
    files["nz_source_2"] = Vector([zs_k7, nz_k7])
    files["nz_source_3"] = Vector([zs_k8, nz_k8])
    files["nz_source_4"] = Vector([zs_k9, nz_k9])

    fake_data = make_theory(files=files);
    fake_data = iΓ * fake_data
    data = fake_data
    folpath = string("../../", method, "_fake_chains/maximization/Y1_3x2_dz_maximization/")
    npzwrite(joinpath(folpath, string("data_", realization,".npz")), data=make_theory())
    println(string("Written data for ", realization,"!"))

    # Create the conditioned model with the dz values fixed.
    cond_model = model(data;
        files=files,
        dz_lens_0=dz_0,
        dz_lens_1=dz_1,
        dz_lens_2=dz_2,
        dz_lens_3=dz_3,
        dz_lens_4=dz_4,
        dz_source_0=dz_5,
        dz_source_1=dz_6,
        dz_source_2=dz_7,
        dz_source_3=dz_8,
        dz_source_4=dz_9)

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
    npzwrite(joinpath(folpath, string("data_", realization+1,".npz")), data=make_theory())
    println(string("Done with chain ", realization+1,"!"))
end