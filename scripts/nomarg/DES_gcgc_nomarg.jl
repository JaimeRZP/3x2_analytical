using LinearAlgebra
using Turing
using LimberJack
using CSV
using YAML
using PythonCall
sacc = pyimport("sacc");

#println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

sacc_path = "../../data/FD/cls_FD_covG.fits"
yaml_path = "../../data/DESY1/gcgc.yml"
nz_path = "../../data/DESY1/nzs"
sacc_file = sacc.Sacc().load_fits(sacc_path)
yaml_file = YAML.load_file(yaml_path)
#nz_DESwl__0 = npzread(string(nz_path, "nz_DESwl__0.npz"))
#nz_DESwl__1 = npzread(string(nz_path, "nz_DESwl__1.npz"))
#nz_DESwl__2 = npzread(string(nz_path, "nz_DESwl__2.npz"))
#nz_DESwl__3 = npzread(string(nz_path, "nz_DESwl__3.npz"))
meta, files = make_data(sacc_file, yaml_file)
                        #nz_DESwl__0=nz_DESwl__0,
                        #nz_DESwl__1=nz_DESwl__1,
                        #nz_DESwl__2=nz_DESwl__2,
                        #nz_DESwl__3=nz_DESwl__3)

data = meta.data
cov = meta.cov

Γ = sqrt(cov)
iΓ = inv(Γ)
data = iΓ * data

init_params=[0.30, 0.05, 0.67, 0.81, 0.95]

@model function model(data;
    meta=meta, 
    files=files)

    #KiDS priors
    Ωm ~ Uniform(0.2, 0.6)
    Ωb ~ Uniform(0.028, 0.065)
    h ~ Truncated(Normal(0.72, 0.05), 0.64, 0.82)
    σ8 ~ Uniform(0.4, 1.2)
    ns ~ Uniform(0.84, 1.1)

    DESgc__0_b = 1.9
    DESgc__1_b = 1.9
    DESgc__2_b = 1.9
    DESgc__3_b = 1.9
    DESgc__4_b = 1.9

    nuisances = Dict("DESgc__0_b" => DESgc__0_b,
                     "DESgc__1_b" => DESgc__1_b,
                     "DESgc__2_b" => DESgc__2_b,
                     "DESgc__3_b" => DESgc__3_b,
                     "DESgc__4_b" => DESgc__4_b)

    cosmology = Cosmology(Ωm=Ωm,  Ωb=Ωb, h=h, ns=ns, σ8=σ8,
        tk_mode=:EisHu,
        pk_mode=:Halofit)

    theory = Theory(cosmology, meta, files; Nuisances=nuisances)
    data ~ MvNormal(theory, cov)
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
folpath = "../../chains/nomarg/"
folname = string("DES_gcgc_nomarg_TAP_", TAP,  "_init_ϵ_", init_ϵ)
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
                init_params=init_params_DES,
                progress=true, save_state=true)

# Save the actual chain.       
@save joinpath(folname, string("chain_", last_n+1,".jls")) chain
CSV.write(joinpath(folname, string("chain_", last_n+1,".csv")), chain)
CSV.write(joinpath(folname, string("summary_", last_n+1,".csv")), describe(chain)[1])