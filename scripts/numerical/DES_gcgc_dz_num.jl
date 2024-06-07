using LinearAlgebra
using Turing
using LimberJack
using CSV
using YAML
using JLD2
using NPZ
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
meta, files = make_data(sacc_file, yaml_file;
                        nz_DESgc__0=nz_DESgc__0,
                        nz_DESgc__1=nz_DESgc__1,
                        nz_DESgc__2=nz_DESgc__2,
                        nz_DESgc__3=nz_DESgc__3,
                        nz_DESgc__4=nz_DESgc__4)

data = meta.data
cov = meta.cov

Γ = sqrt(cov)
iΓ = inv(Γ)
data = iΓ * data

init_params=[0.30, 0.5, 0.67, 0.81, 0.95,
            0.0, 0.0, 0.0, 0.0, 0.0]

@model function model(data;
    meta=meta, 
    files=files)

    #KiDS priors
    Ωm ~ Uniform(0.2, 0.6)
    Ωbb ~ Uniform(0.28, 0.65)
    Ωb := 0.1*Ωbb
    h ~ Truncated(Normal(0.72, 0.05), 0.64, 0.82)
    σ8 ~ Uniform(0.4, 1.2)
    ns ~ Uniform(0.84, 1.1)

    DESgc__0_a ~ Normal(0.0, 1.0) 
    DESgc__1_a ~ Normal(0.0, 1.0)
    DESgc__2_a ~ Normal(0.0, 1.0)
    DESgc__3_a ~ Normal(0.0, 1.0)
    DESgc__4_a ~ Normal(0.0, 1.0)

    DESgc__0_dz := 0.007 * DESgc__0_a
    DESgc__1_dz := 0.007 * DESgc__1_a
    DESgc__2_dz := 0.006 * DESgc__2_a
    DESgc__3_dz := 0.01  * DESgc__3_a
    DESgc__4_dz := 0.01  * DESgc__4_a

    nuisances = Dict("DESgc__0_b" => 1.484,
                     "DESgc__1_b" => 1.805,
                     "DESgc__2_b" => 1.776,
                     "DESgc__3_b" => 2.168,
                     "DESgc__4_b" => 2.23,
                     "DESgc__0_dz" => DESgc__0_dz,
                     "DESgc__1_dz" => DESgc__1_dz,
                     "DESgc__2_dz" => DESgc__2_dz,
                     "DESgc__3_dz" => DESgc__3_dz,
                     "DESgc__4_dz" => DESgc__4_dz,
                     "DESwl__0_m" => 0.018,
                     "DESwl__1_m" => 0.014,
                     "DESwl__2_m" => 0.01,
                     "DESwl__3_m" => 0.004,
                     "A_IA" => 0.294,
                     "alpha_IA" => 0.378)

    cosmology = Cosmology(Ωm=Ωm, Ωb=Ωb, h=h, ns=ns, σ8=σ8,
        tk_mode=:EisHu,
        pk_mode=:Halofit)

    nui_type = eltype(DESgc__0_dz)
    if cosmology.settings.cosmo_type == Float64 && nui_type != Float64
        cosmology.settings.cosmo_type = nui_type
    end

    theory := Theory(cosmology, meta, files; Nuisances=nuisances)
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
folpath = "../../chains_right_nzs/numerical/"
folname = string("DES_gcgc_dz_num_TAP_", TAP)
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
sampler = Gibbs(
        NUTS(adaptation, TAP,
        :Ωm, :Ωbb, :h, :σ8, :ns),
        NUTS(adaptation, TAP,
        :DESgc__0_a, :DESgc__1_a, :DESgc__2_a, :DESgc__3_a, :DESgc__4_a))
chain = sample(cond_model, sampler, iterations;
                init_params=init_params,
                progress=true, save_state=true)

# Save the actual chain.       
@save joinpath(folname, string("chain_", last_n+1,".jls")) chain
CSV.write(joinpath(folname, string("chain_", last_n+1,".csv")), chain)
CSV.write(joinpath(folname, string("summary_", last_n+1,".csv")), describe(chain)[1])
