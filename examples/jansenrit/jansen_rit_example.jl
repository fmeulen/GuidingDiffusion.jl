wdir = @__DIR__
cd(wdir)
outdir= joinpath(wdir, "out")

using Revise
using LinearAlgebra
using GuidingDiffusion
using Distributions
using Bridge
using Plots
using StaticArrays
#import Bridge: constdiff


include("plotting.jl")

################################  TESTING  ################################################

S = DE(Vern7())

include("generatedata.jl")

timegrids = set_timegrids(obs, 0.0005)


iterations = 2_000  #5_00
skip_it = 200
subsamples = 0:skip_it:iterations # for saving paths

# define priors

# prior = Dict("A"=> Exponential(3.25), 
#             "B" => Exponential(22.0),
#             "C" => Exponential(135.0),
#             "α1" => Uniform(0.0, 1.0),
#             "α2" => Uniform(0.0, 1.0),
#             "e0" => Exponential(5.0),
#             "v0" => Exponential(6.0),
#             "r" => Exponential(0.56),
#             "μ" => Exponential(200.0),
#             "σ" => Uniform(0.0, 10_000.0)  )

Π = (A=Exponential(3.25), 
        B=Exponential(22.0),
        C=Exponential(135.0),
        α1=Beta(1.0, 1.0),
        α2=Beta(1.0, 1.0),
        e0=Exponential(5.0),
        v0=Exponential(6.0),
        r=Exponential(0.56),
        μ=Exponential(200.0),
        σ=InverseGamma(0.1, 0.1) )


𝒯 = (A=(short=0.2, long=1.0),
B=(short=0.2, long=1.0),
C=(short=0.2, long=1.0),
α1=(short=0.2, long=1.0),
α2=(short=0.2, long=1.0),
e0=(short=0.2, long=1.0),
v0=(short=0.2, long=1.0),
r=(short=0.2, long=1.0),
μ=(short=0.2, long=1.0),
σ=(short=0.2, long=1.0) )




#import GuidingDiffusion: parameterkernel

# some testing

move = ParMove([:A, :μ], false, 𝒯, Π)
ParMove([:σ], false, 𝒯, Π)

θ = exp.(rand(2))
θᵒ = move.K(θ)

Πsub = [getfield(Π,x) for x in [:A, :μ] ] 

moveσ = ParMove([:σ], parameterkernel((short=[3.0], long=[10.0]); s=0.0), priorσ, true)

moveCᵒ = ParMove([:C], parameterkernel((short=[40.0], long=[100.0])), priorC, false)

moveCσ = ParMove([:C, :σ], parameterkernel((short=[2.0, 10.0], long=[10.0, 10.0]); s=0.0), product_distribution([priorC, priorσ]), true)

moveCσα1 = ParMove([:C, :σ, :α1], parameterkernel((short=[2.0, 10.0, 0.01], long=[10.0, 10.0, 0.1]); s=0.0), product_distribution([priorC, priorσ, priorα1]), true)

moveCα1 = ParMove([:C, :α2], parameterkernel((short=[2.0,  0.01], long=[10.0,  0.1]); s=0.3), product_distribution([priorC,  priorα1]), false)
# a small program

# settings
verbose = true # if true, surpress output written to console


θinit = 100.0
ESTσ = true



if ESTσ
  θ = (C=copy(θinit), σ = 2.0)
  movetarget = moveCσ
  allparnames = [:C, :σ]
else
  θ = (; C = copy(θinit) ) # initial value for parameter
  movetarget = moveC
  allparnames = [:C]
end
ℙ = setproperties(ℙ0, θ)

θ = (C=copy(θinit), α1=0.5)
movetarget = moveCα1
allparnames = [:C,  :α2]
ℙ = setproperties(ℙ0, θ)

𝒯 = 40.0 # temperature
ℙe = setproperties(ℙ0, C=copy(θinit),  σ = 𝒯)
allparnamese = [:C]
move_exploring = moveCᵒ



# pcn pars 
ρ = 0.95
ρe = 0.95

# initialisation of target chain 
B = BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids);
Z = Innovations(timegrids, ℙ);
Zbuffer = deepcopy(Z)
Zᵒ = deepcopy(Z)
ρs = fill(ρ, length(timegrids))
XX, ll = forwardguide(B, ℙ)(x0, Z);


# initialisation of exploring chain 
Be = BackwardFilter(S, ℙe, AuxType, obs, obsvals, timegrids);
Ze = Innovations(timegrids, ℙe);# deepcopy(Z);
Zeᵒ = deepcopy(Ze)
ρse = fill(ρe, length(timegrids))
XXe, lle = forwardguide(Be, ℙe)(x0, Ze);



θsave = [copy(getpar(allparnames, ℙ))]
XXsave = [copy(XX)]
llsave = [ll]

XXesave = [copy(XXe)]



accinnov = 0
accpar = 0
accinnove = 0
accpare = 0
accmove = 0


exploring = [State(x0, copy(Ze), getpar(allparnamese,ℙe), copy(lle))]

for i in 1:iterations
  (i % 500 == 0) && println(i)
  
  # update exploring chain
  if i==1  
  lle, Be, ℙe, accpare_ = parupdate!(Be, XXe, move_exploring, obs, obsvals, S, AuxType, timegrids; verbose=verbose)(x0, ℙe, Ze, lle);# θe and XXe may get overwritten
  lle, accinnove_ = pcnupdate!(Be, ℙe, XXe, Zbuffer, Zeᵒ, ρse)(x0, Ze, lle); # Z and XX may get overwritten
  push!(exploring, State(x0, copy(Ze), getpar(allparnamese, ℙe), copy(lle)))   # collection of samples from exploring chain
  end

  # update target chain
  smallworld = rand() > 0.33
  if smallworld
    ll, B, ℙ, accpar_ = parupdate!(B, XX, movetarget, obs, obsvals, S, AuxType, timegrids; verbose=verbose)(x0, ℙ, Z, ll);# θ and XX may get overwritten
    accmove_ =0
  else
    w = sample(exploring)     # randomly choose from samples of exploring chain
    ll, ℙ,  accmove_ = exploremoveσfixed!(B, Be, ℙe, move_exploring, XX, Zᵒ, w; verbose=verbose)(x0, ℙ, Z, ll) 
    accpar_ = 0
    #println(ℙ.C ==θ[1])
  end  
  ll, accinnov_ = pcnupdate!(B, ℙ, XX, Zbuffer, Zᵒ, ρs)(x0, Z, ll); # Z and XX may get overwritten

  
  # update acceptance counters
#  accpar += accpar_; accpare += accpare_; accinnove += accinnove_; accinnov += accinnov_; accmove += accmove_
  # saving iterates
  push!(θsave, getpar(allparnames, ℙ))
  push!(llsave, ll)
  (i in subsamples) && push!(XXsave, copy(XX))
  (i % 500 == 0) && push!(XXesave, XXe)
  
  adjust_PNCparamters!(ρs, ρ)
end




# final imputed path
plot_all(ℙ, timegrids, XXsave[end])

obstimes = getfield.(obs, :t)

plot_all(ℙ, Xf, obstimes, obsvals, timegrids, XXsave[1])
savefig(joinpath(outdir,"guidedpath_firstiteration.png"))
plot_all(ℙ, Xf, obstimes, obsvals, timegrids, XXsave[end])
savefig(joinpath(outdir,"guidedpath_finaliteration.png"))

plot_all(ℙ, Xf, obstimes, obsvals, timegrids, XXesave[1])
savefig(joinpath(outdir,"guidedpath_firstiteration_exploring.png"))
plot_all(ℙ, Xf, obstimes, obsvals, timegrids, XXesave[end])
savefig(joinpath(outdir,"guidedpath_finaliteration_exploring.png"))


#
println("Target chain: accept% innov ", 100*accinnov/iterations,"%")
println("Target chain: accept% par ", 100*accpar/iterations,"%")
println("Exploring chain: accept% innov ", 100*accinnove/iterations,"%")
println("Exploring chain: accept% par ", 100*accpare/iterations,"%")

println("accept% swap ", 100*accmove/iterations,"%")

θesave = getindex.(getfield.(exploring,:θ),1)
llesave = getfield.(exploring, :ll)

h1 = histogram(getindex.(θsave,1),bins=35, label="target chain")
h2 = histogram(getindex.(θesave,1),bins=35, label="exploring chain")
plot(h1, h2, layout = @layout [a b])  

p1 = plot(llsave, label="target",legend=:bottom)    
plot!(p1,llesave, label="exploring")    
savefig(joinpath(outdir,"logliks.png"))

# traceplots
pa = plot(getindex.(θsave,1), label="target", legend=:top)
hline!(pa, [ℙ0.C], label="",color=:black)
plot!(pa, getindex.(θesave,1), label="exploring")

if ESTσ 
  pb = plot(getindex.(θsave,2), label="target", legend=:top)
  hline!(pb, [ℙ0.σ], label="",color=:black)
  plot(pa, pb, layout = @layout [a; b])  
end
savefig(joinpath(outdir,"traceplots.png"))


scatter(getindex.(θsave,1), getindex.(θsave,2))



