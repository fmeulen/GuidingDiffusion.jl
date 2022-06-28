# first illustrate bimodality of the likelihood
Cgrid = 5.0:0.5:400.0
N_mc = 25
Zs = [Innovations(timegrids, ℙ0)]
for j=2:N_mc
  push!(Zs, Innovations(timegrids, ℙ0))
end

function loglik(B, ℙ0, x0, Cval, Zs)
  ℙ = setproperties(ℙ0, C=Cval)
  s = 0.0
  for Z in Zs
    _, ll = forwardguide(B, ℙ)(x0, Z);
    s += ll
  end
  s/length(Zs)
end

llC = zeros(length(Cgrid))
for k ∈ eachindex(Cgrid)
    println(Cgrid[k])
    llC[k] = loglik(B, ℙ0, x0, Cgrid[k], Zs)
end

pa = plot(Cgrid, llC, label="loglik"); vline!([ℙ0.C], label="true",legend=:bottomright)
half_ind = div(length(Cgrid),2)
pb = plot(Cgrid[1:half_ind], llC[1:half_ind], label=""); vline!([ℙ0.C], label="")
lay = @layout [a b]
#pbimod = plot(pa, pb, layout=lay, xlabel="C", ylabel="loglikelihood",size = (700, 300)) 

using Optim

negloglik(B, ℙ0, x0, Zs) = (C) -> -loglik(B, ℙ0, x0, C, Zs)
res = optimize(negloglik(B, ℙ0, x0, Zs), 0.0, 500.0)   
print(res)

vline!(pa, [res.minimizer], label="mle")
vline!(pb, [res.minimizer], label="")
pbimod = plot(pa, pb, layout=lay, xlabel="C", ylabel="loglikelihood",size = (700, 300)) 

savefig(out*"/generated/bimodalityplot_C.png")

μgrid = 5.0:2.0:400.0
llμ = zeros(length(μgrid))
for k ∈ eachindex(μgrid)
  ℙ = setproperties(ℙ0, μ=μgrid[k])
  println(ℙ.μ)
  s = 0.0
  for j in 1:N_mc
    _, ll = forwardguide(B, ℙ)(x0, Zs[j]);
    s += ll
  end
  #push!(llC, copy(ll))
 llμ[k] = s/N_mc
end
pa = plot(μgrid, llμ, label="loglik"); vline!([ℙ0.μ], label="true val of μ",legend=:bottomright)
pbimod = plot(pa, xlabel="μ", ylabel="loglikelihood") 

savefig(joinpath(outdir,"bimodalityplot_mu.png"))
