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


iterations = 10_00  #5_00
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


𝒯 = (A=(short=0.02, long=.1),
B=(short=0.2, long=1.0),
C=(short=0.01, long=.5),
α1=(short=0.01, long=.1),
α2=(short=0.02, long=.1),
e0=(short=0.2, long=1.0),
v0=(short=0.2, long=1.0),
r=(short=0.2, long=1.0),
μ=(short=0.2, long=1.0),
σ=(short=0.2, long=1.0) )




#import GuidingDiffusion: parameterkernel

# some testing
params = [:C, :α1]#, :B]

moves = repeat([ParMove([:C], false, 𝒯, Π),
         ParMove([:α1], false, 𝒯, Π)], 3)
push!(moves, ParMove([:C, :α1], true, 𝒯, Π))

moves = [ParMove([:C], false, 𝒯, Π),ParMove([:α1], true, 𝒯, Π)]

params = [:α1]
moves = [ParMove([:α1], true, 𝒯, Π)]

        #  ,
        #  ParMove([:B], false, 𝒯, Π)]
#,ParMove([:C, :α1], false, 𝒯, Π)]

# very simple check case
# params = [:C]
# moves = [ParMove([:C], false, 𝒯, Π)]



# moveσ = ParMove([:σ], parameterkernel((short=[3.0], long=[10.0]); s=0.0), priorσ, true)
# moveCᵒ = ParMove([:C], parameterkernel((short=[40.0], long=[100.0])), priorC, false)
# moveCσ = ParMove([:C, :σ], parameterkernel((short=[2.0, 10.0], long=[10.0, 10.0]); s=0.0), product_distribution([priorC, priorσ]), true)
# moveCσα1 = ParMove([:C, :σ, :α1], parameterkernel((short=[2.0, 10.0, 0.01], long=[10.0, 10.0, 0.1]); s=0.0), product_distribution([priorC, priorσ, priorα1]), true)
# moveCα1 = ParMove([:C, :α2], parameterkernel((short=[2.0,  0.01], long=[10.0,  0.1]); s=0.3), product_distribution([priorC,  priorα1]), false)
# a small program

# settings
verbose = true # if true, surpress output written to console


# θinit = 50.0
# ESTσ = false



# if ESTσ
#   θ = (C=copy(θinit), σ = 2.0)
#   movetarget = move
#   allparnames = [:C, :σ]
# else
#   θ = (; C = copy(θinit) ) # initial value for parameter
#   movetarget = move
#   allparnames = [:C]
# end

# initialise parameter
ℙ = setproperties(ℙ0,  α1 = 0.91)
temp = 40.0 # temperature
ℙe = setproperties(ℙ0,   σ = temp, α1 = 0.1)



# overwrite, figure out later precisely
# allparnames = params
# allparnamese = params

# pcn pars 
ρ = 0.98
ρe = 0.98

# initialisation of target chain 
B = BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids);
Z = Innovations(timegrids, ℙ);
Zbuffer = deepcopy(Z)
Zᵒ = deepcopy(Z)
ρs = fill(ρ, length(timegrids))
XX, ll = forwardguide(B, ℙ)(x0, Z);


# initialisation of exploring chain 
Be = BackwardFilter(S, ℙe, AuxType, obs, obsvals, timegrids);
Ze = deepcopy(Z)# Innovations(timegrids, ℙe);# deepcopy(Z);
Zeᵒ = deepcopy(Ze)
ρse = fill(ρe, length(timegrids))
XXe, lle = forwardguide(Be, ℙe)(x0, Ze);



θsave = [copy(getpar(params, ℙ))]
XXsave = [copy(XX)]
llsave = [ll]

XXesave = [copy(XXe)]

accinnov = 0
accpar = 0
accinnove = 0
accpare = 0
accmove = 0

allparnamese = params


exploring = [State(x0, copy(Ze), getpar(params,ℙe), copy(lle))]

accpar_= accpare_ = accinnove_= accinnov_= accmove_ =0
for i in 1:iterations
  (i % 500 == 0) && println(i)
  global accpar_, accpare_,accinnove_, accinnov_, accmove_
  
  for move ∈ moves
    # update exploring chain
    lle, Be, ℙe, accpare_ = parupdate!(Be, XXe, move, obs, obsvals, S, AuxType, timegrids; verbose=verbose)(x0, ℙe, Ze, lle);
    lle, accinnove_ = pcnupdate!(Be, ℙe, XXe, Zbuffer, Zeᵒ, ρse)(x0, Ze, lle); 
      
    # update target chain   
    smallworld = rand() > 0#.33
    if smallworld
      ll, B, ℙ, accpar_ = parupdate!(B, XX, move, obs, obsvals, S, AuxType, timegrids; verbose=verbose)(x0, ℙ, Z, ll);
      accmove_ =0
    else
      w = sample(exploring)     # randomly choose from samples of exploring chain
      ll, ℙ,  accmove_ = exploremoveσfixed!(B, Be, ℙe, move, XX, Zᵒ, w; verbose=verbose)(x0, ℙ, Z, ll) 
      accpar_ = 0
    end  
    ll, accinnov_ = pcnupdate!(B, ℙ, XX, Zbuffer, Zᵒ, ρs)(x0, Z, ll);
  end

  # update exploring
  push!(exploring, State(x0, copy(Ze), getpar(params, ℙe), copy(lle)))   # collection of samples from exploring chain

  # update acceptance counters
  accpar += accpar_; accpare += accpare_; accinnove += accinnove_; accinnov += accinnov_; accmove += accmove_
  # saving iterates
  push!(θsave, getpar(params, ℙ))
  push!(llsave, ll)
  (i in subsamples) && push!(XXsave, copy(XX))
  (i % 500 == 0) && push!(XXesave, XXe)
  
 # adjust_PCNparameters!(ρs, ρ)
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

θesave = getfield.(exploring,:θ)
llesave = getfield.(exploring, :ll)

h1 = histogram(getindex.(θsave,1),bins=35, label="target chain")
h2 = histogram(getindex.(θesave,1),bins=35, label="exploring chain")
plot(h1, h2, layout = @layout [a b])  

p1 = plot(llsave, label="target",legend=:bottom)    
plot!(p1,llesave, label="exploring")    
savefig(joinpath(outdir,"logliks.png"))

# traceplots
i = 1
p1 = plot(getindex.(θsave,i), label="target", legend=:top)
hline!(p1, [getfield(ℙ0,params[i])], label="",color=:black)
plot!(p1, getindex.(θesave,i), label="exploring")

@error "stop"

i = 2
p2 = plot(getindex.(θsave,i), label="target", legend=:top)
hline!(p2, [getfield(ℙ0,params[i])], label="",color=:black)
plot!(p2, getindex.(θesave,i), label="exploring")

# i = 3
# p3 = plot(getindex.(θsave,i), label="target", legend=:top)
# hline!(p3, [getfield(ℙ0,params[i])], label="",color=:black)
# plot!(p3, getindex.(θesave,i), label="exploring")

plot(p1, p2, layout = @layout [a; b])  




ESTσ = false
if ESTσ 
  pb = plot(getindex.(θsave,2), label="target", legend=:top)
  hline!(pb, [ℙ0.σ], label="",color=:black)
  plot(pa, pb, layout = @layout [a; b])  
end
savefig(joinpath(outdir,"traceplots.png"))


scatter(getindex.(θsave,1), getindex.(θsave,2))



