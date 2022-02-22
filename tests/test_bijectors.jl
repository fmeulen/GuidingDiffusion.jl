function parameterkernel(θ, tuningpars, s) 
    shortrange = rand()>s
    Δ = shortrange ?  rand(MvNormal(tuningpars.short)) : rand(MvNormal(tuningpars.long))
    θ + Δ
  end
  
parameterkernel(tuningpars; s=0.33) = (θ) -> parameterkernel(θ, tuningpars, s) 


parameterkernel((short=[0.2], long=[1.0]), Beta(2,2))([0.2])