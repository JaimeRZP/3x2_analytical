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

sacc_path = "../../data/CosmoDC2/summary_statistics_fourier_tjpcov.sacc"
yaml_path = "../../data/CosmoDC2/gcgc_gcwl_wlwl.yml"
nz_path = "../../data/CosmoDC2/image_nzs/"
sacc_file = sacc.Sacc().load_fits(sacc_path)
yaml_file = YAML.load_file(yaml_path)
#nz_DESwl__0 = npzread(string(nz_path, "nz_DESwl__0.npz"))
#nz_DESwl__1 = npzread(string(nz_path, "nz_DESwl__1.npz"))
#nz_DESwl__2 = npzread(string(nz_path, "nz_DESwl__2.npz"))
#nz_DESwl__3 = npzread(string(nz_path, "nz_DESwl__3.npz"))
meta, files = make_data(sacc_file, yaml_file)#;
                        #nz_DESwl__0=nz_DESwl__0,
                        #nz_DESwl__1=nz_DESwl__1,
                        #nz_DESwl__2=nz_DESwl__2,
                        #nz_DESwl__3=nz_DESwl__3);

meta_3x2.types = [ 
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

data = meta.data
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

    nuisances = Dict{String, Float64}(
        "lens_0_b"    => 0.879118,
        "lens_1_b"    => 1.05894,
        "lens_2_b"    => 1.22145,
        "lens_3_b"    => 1.35065,
        "lens_4_b"    => 1.58909,
        "lens_0_dz"   => -0.00430633,
        "lens_1_dz"   => -0.00316143,
        "lens_2_dz"   => 0.00459595,
        "lens_3_dz"   => -0.00148823,
        "lens_4_dz"   => -0.00150412,
        "source_0_m"  => -0.00733846,
        "source_1_m"  => -0.00434667,
        "source_2_m"  => 0.00434908,
        "source_3_m"  => -0.00278755,
        "source_4_m"  => 0.000101118,
        "source_0_dz" => 0.00119753,
        "source_1_dz" => 0.00195125,
        "source_2_dz" => -0.000356315,
        "source_3_dz" => 0.00175369,
        "source_4_dz" => 0.00379481)
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
folpath = "../../chains_right_nzs/nomarg/"
folname = string("CosmoDC2_3x2_nomarg_TAP_", TAP)
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
