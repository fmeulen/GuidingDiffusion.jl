θ0 =[3.25, 100.0, 22.0, 50.0, 190.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 6.0]  # except for μy as in Buckwar/Tamborrino/Tubikanec#
ℙ0 = JansenRitDiffusion(θ0...)
@show properties(ℙ0)
AuxType = JansenRitDiffusionAux

T =5.0
x00 = @SVector zeros(6)
W = sample((-1.0):0.0001:T, wienertype(ℙ0))                        #  sample(tt, Wiener{ℝ{1}}())
Xf_prelim = solve(Euler(), x00, W, ℙ0)
# drop initial nonstationary behaviour
Xf = SamplePath(Xf_prelim.tt[10001:end], Xf_prelim.yy[10001:end])
x0 = Xf.yy[1]
dt = Xf.tt[2]-Xf.tt[1]

skipobs = 40# I took 400  all the time
obstimes =  Xf.tt[1:skipobs:end]
obsvals = map(x -> L*x, Xf.yy[1:skipobs:end])
pF = plot_all(ℙ0,  Xf, obstimes, obsvals)


#------- process observations, assuming x0 known
obs = [Observation(obstimes[1],  x0,  SMatrix{6,6}(1.0I), SMatrix{6,6}(Σdiagel*I))]
for i in 2:length(obstimes)
  push!(obs, Observation(obstimes[i], obsvals[i], L, Σ));
end


timegrids = set_timegrids(obs, 0.00005)
B = BackwardFilter(S, ℙ0, AuxType, obs, obsvals, timegrids) ;
Z = Innovations(timegrids, ℙ0);
XX, ll = forwardguide(B, ℙ0)(x0, Z);
plot_all(ℙ0,Xf,  obstimes, obsvals, timegrids, XX)

llC =[]
Cgrid = 5.0:2.5:400.0
for C ∈ Cgrid
  ℙ = setproperties(ℙ0, C=C)
  println(ℙ.C)
  _, ll = forwardguide(B, ℙ)(x0, Z);
  push!(llC, copy(ll))
end
plot(Cgrid, llC, label="loglik"); vline!([ℙ0.C], label="true val of C")

llμ =[]
μgrid = 5.0:2.0:400.0

B = BackwardFilter(S, ℙ0, AuxType, obs, obsvals, timegrids) ;
for μ ∈ μgrid
  ℙ = setproperties(ℙ0, μ=μ)
  _, ll = forwardguide(B, ℙ)(x0, Z);
  push!(llμ, copy(ll))
end
plot(μgrid, llμ, label="loglik"); vline!([ℙ0.μ], label="true val of μ")


llA =[]
Agrid = .2:.2:10.0
B = BackwardFilter(S, ℙ0, AuxType, obs, obsvals, timegrids) ;
for A ∈ Agrid
  ℙ = setproperties(ℙ0, A=A)
  #B = BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids) ;
  _, ll = forwardguide(B, ℙ)(x0, Z);
  push!(llA, copy(ll))
end
plot(Agrid, llA, label="loglik"); vline!([ℙ0.A], label="true val of A")
plot(Agrid[10:end], llA[10:end], label="loglik"); vline!([ℙ0.A], label="true val of A")


llσ =[]
σgrid = 1.0:.2:42.0
#B = BackwardFilter(S, ℙ0, AuxType, obs, obsvals, timegrids) ;
for σ ∈ σgrid
  ℙ = setproperties(ℙ0, σ=σ)
  B = BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids) ;
  _, ll = forwardguide(B, ℙ)(x0, Z);
  push!(llσ, copy(ll))
end
plot(σgrid, llσ, label="loglik")
vline!([ℙ0.σ], label="true val of σy")

 llv0 = []
 B = BackwardFilter(S, ℙ0, AuxType, obs, obsvals, timegrids) ;
 v0grid = .1:.2:18.0
 for v0 ∈ v0grid
   ℙ = setproperties(ℙ0, v0=v0)
   
   _, ll = forwardguide(B, ℙ)(x0, Z);
   push!(llv0, copy(ll))
 end
 plot(v0grid, llv0, label="loglik")
  vline!([ℙ0.v0], label="true val of νmax")
 


  llα1 = []
  B = BackwardFilter(S, ℙ0, AuxType, obs, obsvals, timegrids) ;
  α1grid = .01:.02:0.99
  for α1 ∈ α1grid
    ℙ = setproperties(ℙ0, α1=α1)

    _, ll = forwardguide(B, ℙ)(x0, Z);
    push!(llα1, copy(ll))
  end
  plot(α1grid, llα1, label="loglik")
   vline!([ℙ0.α1], label="true val of α1")
  

   llα2 = []
   B = BackwardFilter(S, ℙ0, AuxType, obs, obsvals, timegrids) ;
   α2grid = .01:.02:0.99
   for α2 ∈ α2grid
     ℙ = setproperties(ℙ0, α2=α2)
  
     _, ll = forwardguide(B, ℙ)(x0, Z);
     push!(llα2, copy(ll))
   end
   plot(α2grid, llα2, label="loglik")
    vline!([ℙ0.α2], label="true val of α2")
 


Cgrid = 5.0:5.0:400.0
σgrid = 100.0:100.0: 2000.0
llCσ = zeros(length(Cgrid), length(σgrid))
for i ∈ eachindex(Cgrid)
    println(i)
    for j ∈ eachindex(σgrid)
        ℙ = setproperties(ℙ0, C=Cgrid[i], σ = σgrid[j])
        B = BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids) ;
        #println(ℙ.C)
        _, ll = forwardguide(B, ℙ)(x0, Z);
        llCσ[i,j] = ll
    end
end
plot(Cgrid, llC, label="loglik"); vline!([ℙ0.C], label="true val of C")

heatmap(σgrid, Cgrid,  llCσ)
heatmap(llCσ)
ℙ0.C
ℙ0.σ

plot(σgrid, llCσ[30,:])





# should be possible to compute the MLE
parnames = [:A, :α1, :C]


function loglik(ℙ0, θ, x0, Z, parnames,  S, AuxType, obs, obsvals, timegrids)
    tup = (; zip(parnames, θ)...) # try copy here
    ℙ = setproperties(ℙ0, tup)
    B = BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids) ;
    _, ll = forwardguide(B, ℙ)(x0, Z);
    ll
end

loglik(ℙ0, x0, Z, parnames,  S, AuxType, obs, obsvals, timegrids) = (θ) -> loglik(ℙ0, θ, x0, Z, parnames,  S, AuxType, obs, obsvals, timegrids)
 
using Optim
getpar(parnames, ℙ0)

θ = [4.0, 0.4, 220.0]
lower = [0.5, 0.1, 50.0]
upper = [10.0, 0.9, 250.0]

Optim.optimize(loglik(ℙ0, x0, Z, parnames,  S, AuxType, obs, obsvals, timegrids), lower, upper, θ, SimulatedAnnealing())

ForwardDiff.gradient(loglik(ℙ0, x0, Z, parnames,  S, AuxType, obs, obsvals, timegrids), lower, upper, θ)

Zygote.gradient(loglik(ℙ0, x0, Z, parnames,  S, AuxType, obs, obsvals, timegrids),  θ)
Yota.grad(loglik(ℙ0, x0, Z, parnames,  S, AuxType, obs, obsvals, timegrids),  θ)
# drawing from the priors for initialisation 


# test 
fff(x) = sum(sin.(cos.(x)))
fff(rand(5))

x_eval = rand(500)
Zygote.gradient(fff, x_eval)
Yota.grad(fff, x_eval)


# further testing

function loglik(θ, x0, Z, S, AuxType, obs, obsvals, timegrids)
  θp =[θ[1], θ[2], 22.0, 50.0, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 2000.0]  # except for μy as in Buckwar/Tamborrino/Tubikanec#
  ℙ = JansenRitDiffusion(θp...)
    B = BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids) ;
  _, ll = forwardguide(B, ℙ)(x0, Z);
  ll
end

loglik(x0,Z,  S, AuxType, obs, obsvals, timegrids) = (θ) -> loglik(θ, x0, Z,  S, AuxType, obs, obsvals, timegrids)

θ = [50.0, 100.0]
Zygote.gradient(loglik(x0, Z,  S, AuxType, obs, obsvals, timegrids),  θ)
Yota.grad(loglik(x0, Z,  S, AuxType, obs, obsvals, timegrids),  θ)

ff = (Z)-> forwardguide(B, ℙ)(x0, Z)[2]
ff(Z)
Zygote.gradient(ff, Z)
Yota.grad(ff, Z)


logh̃(x, h0) =  dot(x, -0.5 * h0.H * x + h0.F) - h0.C    

function orwardguide(x0, ℙ, Z, M::Message)
    ℙ̃ = M.ℙ̃
    tt = M.tt
    X = [x0]
    x = x0
    ll::eltype(x0)  = 0.
    for i ∈ 1:length(tt)-1
        dt = tt[i+1]-tt[i]
        b = Bridge.b(tt[i], x, ℙ) 
        r = M.F[i] - M.H[i] * x 
        σ = Bridge.σ(tt[i], x, ℙ)
        dz = Z.yy[i+1] - Z.yy[i]
        
        # likelihood terms
        if i<=length(tt)-1-sk
            db = b  - Bridge.b(tt[i], x, ℙ̃)
            ll += dot(db, r) * dt
            if !Bridge.constdiff(ℙ) || !Bridge.constdiff(ℙ̃)
                σ̃ = Bridge.σ(tt[i], ℙ̃)
                ll += 0.5*Bridge.inner( σ' * r) * dt    # |σ(t,x)' * tilder(t,x)|^2
                ll -= 0.5*Bridge.inner(σ̃' * r) * dt   # |tildeσ(t)' * tilder(t,x)|^2
                a = Bridge.a(tt[i], x, ℙ)
                ã = Bridge.a(tt[i], ℙ̃)
                ll += 0.5*dot(a-ã, M.H[i]) * dt
            end
        end
        x  +=  b * dt + σ* (σ' * r * dt + dz) 
        #push!(X, copy(x))
    end
    X, ll
end



ff = (Z) ->  orwardguide(x0, ℙ, Z.z[1], B.Ms[1])[2]
sk=1
ff(Z)
Zygote.gradient(ff, Z)
Yota.grad(ff,Z)

ff2 = (u) ->  orwardguide(x0, ℙ, u, B.Ms[1])[2]
Zygote.gradient(ff2, Z.z[1])
ForwardDiff.gradient(ff2, Z.z[1])
# does Zygote work with static arrays?
