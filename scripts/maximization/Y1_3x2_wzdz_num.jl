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
param_path = string("../../data/CosmoDC2/nzs_", method, "/wzdz_priors/wzdz_params.npz")

sacc_file = sacc.Sacc().load_fits(sacc_path)
yaml_file = YAML.load_file(yaml_path)
wzdz_params = npzread(param_path)

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

for realization in 1:10_000

    wzdz_0 = wzdz_params["lens_0"][:, realization]
    wzdz_1 = wzdz_params["lens_1"][:, realization]
    wzdz_2 = wzdz_params["lens_2"][:, realization]
    wzdz_3 = wzdz_params["lens_3"][:, realization]
    wzdz_4 = wzdz_params["lens_4"][:, realization]
    wzdz_5 = wzdz_params["source_0"][:, realization]
    wzdz_6 = wzdz_params["source_1"][:, realization]
    wzdz_7 = wzdz_params["source_2"][:, realization]
    wzdz_8 = wzdz_params["source_3"][:, realization]
    wzdz_9 = wzdz_params["source_4"][:, realization]

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

    _nz_lens_0 = Dict("z"=>  zs_k0, "dndz"=>  nz_k0)
    _nz_lens_1 = Dict("z"=>  zs_k1, "dndz"=>  nz_k1)
    _nz_lens_2 = Dict("z"=>  zs_k2, "dndz"=>  nz_k2)
    _nz_lens_3 = Dict("z"=>  zs_k3, "dndz"=>  nz_k3)
    _nz_lens_4 = Dict("z"=>  zs_k4, "dndz"=>  nz_k4)
    _nz_source_0 = Dict("z"=>  zs_k5, "dndz"=>  nz_k5)
    _nz_source_1 = Dict("z"=>  zs_k6, "dndz"=>  nz_k6)
    _nz_source_2 = Dict("z"=>  zs_k7, "dndz"=>  nz_k7)
    _nz_source_3 = Dict("z"=>  zs_k8, "dndz"=>  nz_k8)
    _nz_source_4 = Dict("z"=>  zs_k9, "dndz"=>  nz_k9)

    meta, files = make_data(sacc_file, yaml_file;
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

    fake_data = make_theory();
    fake_data = iΓ * fake_data
    data = fake_data
    npzwrite(joinpath(folpath, string("data_", realization+1,".npz")), data=make_theory())
    println(string("Written data for ", realization+1,"!"))

    @model function model(data)
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

        theory := make_theory(Ωm=Ωm, Ωb=Ωb, h=h, σ8=σ8, ns=ns,
                                dz_lens_0=wzdz_0[1], wz_lens_0=wzdz_0[2],
                                dz_lens_1=wzdz_1[1], wz_lens_1=wzdz_1[2],
                                dz_lens_2=wzdz_2[1], wz_lens_2=wzdz_2[2],
                                dz_lens_3=wzdz_3[1], wz_lens_3=wzdz_3[2],
                                dz_lens_4=wzdz_4[1], wz_lens_4=wzdz_4[2],
                                dz_source_0=wzdz_5[1], wz_source_0=wzdz_5[2],
                                dz_source_1=wzdz_6[1], wz_source_1=wzdz_6[2],
                                dz_source_2=wzdz_7[1], wz_source_2=wzdz_7[2],
                                dz_source_3=wzdz_8[1], wz_source_3=wzdz_8[2],
                                dz_source_4=wzdz_9[1], wz_source_4=wzdz_9[2],
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

    cond_model = model(data)

    # Find MLE estimate to use as initial parameters.
    mle = maximum_likelihood(cond_model, NelderMead())
    values = mle.values.array
    namess = mle.values.dicts[1].keys

    folpath = string("../../", method, "_fake_chains/maximization/Y1_3x2_wzdz_naximization/")
    filename = string("samples.csv")
    params = Dict{Symbol, Float64}()
    for (i, name) in enumerate(namess)
        params[Symbol(name)] = values[i]
    end
    if realization == 0
        CSV.write(joinpath(folpath, filename), DataFrame(params))
    else
        CSV.write(joinpath(folpath, filename), DataFrame(params); append=true)
    end
    npzwrite(joinpath(folpath, string("data_", realization+1,".npz")), data=make_theory())
    println(string("Done with chain ", realization+1,"!"))
end