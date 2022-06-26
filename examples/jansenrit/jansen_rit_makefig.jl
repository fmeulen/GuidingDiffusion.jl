

# final imputed path
# plot_all(ℙ, timegrids, XXsave[end])
# obstimes = getfield.(obs, :t)

# intial guided path
plot_all(ℙ, Xf, obstimes, obsvals, timegrids, XXsave[1])
savefig(joinpath(outdir,"guidedpath_firstiteration.png"))
plot_all(ℙ, Xf, obstimes, obsvals, timegrids, XXsave[end])
savefig(joinpath(outdir,"guidedpath_finaliteration.png"))

# guided path in last iteration
plot_all(ℙ, Xf, obstimes, obsvals, timegrids, XXesave[1])
savefig(joinpath(outdir,"guidedpath_firstiteration_exploring.png"))
plot_all(ℙ, Xf, obstimes, obsvals, timegrids, XXesave[end])
savefig(joinpath(outdir,"guidedpath_finaliteration_exploring.png"))


# compute acceptance percentages
accperc_innov_target = 100*accinnov/iterations
accperc_par_target = 100*accpar/iterations
accperc_innov_exploring = 100*accinnove/iterations
accperc_par_exploring = 100*accpare/iterations
accperc_swap = 100*accmove/iterations

# histograms of parameters
if swap_allowed
  θesave = getfield.(exploring,:θ)
  llesave = getfield.(exploring, :ll)
  h1 = histogram(getindex.(θsave,1),bins=35, label="target chain")
  h2 = histogram(getindex.(θesave,1),bins=35, label="exploring chain")
  h3 = histogram(getindex.(θsave,2),bins=35, label="target chain")
  h4 = histogram(getindex.(θesave,2),bins=35, label="exploring chain")

  plot(h1, h2, h3, h4, layout = @layout [a b; c d])  
  savefig(joinpath(outdir,"histograms_pars.png"))
else
  h1 = histogram(getindex.(θsave,1),bins=35, label="target chain")
  h3 = histogram(getindex.(θsave,2),bins=35, label="target chain")
  plot(h1, h3, layout = @layout [a b])  
  savefig(joinpath(outdir,"histograms_pars.png"))
end

# likelihood for target chain
p1 = plot(llsave)#, label="target",legend=:bottom)    
#plot!(p1,llesave, label="exploring")    
savefig(joinpath(outdir,"logliks.png"))

# traceplots 
if swap_allowed  #both target and exploring chain
  i = 1
  p1 = plot(getindex.(θsave,i), label="target", legend=:top)
  hline!(p1, [getfield(ℙ0,params[i])], label="",color=:black)
  plot!(p1, getindex.(θesave,i), label="exploring")
  i = 2
  p2 = plot(getindex.(θsave,i), label="target", legend=:top)
  hline!(p2, [getfield(ℙ0,params[i])], label="",color=:black)
  plot!(p2, getindex.(θesave,i), label="exploring")
  plot(p1, p2, layout = @layout [a; b])  
  savefig(joinpath(outdir,"traceplots.png"))
else # only target chain
  i = 1
  p1_ = plot(getindex.(θsave,i), label="target", legend=:top)
  hline!(p1_, [getfield(ℙ0,params[i])], label="",color=:black)
  i = 2
  p2_ = plot(getindex.(θsave,i), label="target", legend=:top)
  hline!(p2_, [getfield(ℙ0,params[i])], label="",color=:black)
  plot(p1_, p2_, layout = @layout [a; b])  
  savefig(joinpath(outdir,"traceplots_target.png"))
end



# Write some info to txt file
open(joinpath(outdir,"info.txt"), "w") do f
  println(f, "params $params"); println(f, "") 
  println(f, "iterations $iterations")
  println(f, "swap_allowed $swap_allowed")
  println(f,"accperc_innov_target $accperc_innov_target")
  println(f,"accperc_par_target $accperc_par_target")
  println(f,"accperc_innov_exploring $accperc_innov_exploring")
  println(f,"accperc_par_exploring $accperc_par_exploring")
  println(f, "accperc_swap $accperc_swap")
  println(f, "")
  println(f, "tuning pars")
  print(f, "$𝒯")
  println(f, "");println(f, "")
  println(f, "prior")
  print(f, "$Π")
  println(f,"")
  println(f, "ρ $ρ")
  println(f, "ρe $ρe")
  println(f,"")
  println(f, "ℙinit $ℙinit")
  println(f, "ℙ0 $ℙ0")
  println(f, "temperature temp  $temp")
end

@error "stop"


ESTσ = false
if ESTσ 
  pb = plot(getindex.(θsave,2), label="target", legend=:top)
  hline!(pb, [ℙ0.σ], label="",color=:black)
  plot(pa, pb, layout = @layout [a; b])  
end


