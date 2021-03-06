Random.seed!(5)


model= [:jr, :jr3][1]

if model == :jr
  θ0 =[3.25, 100.0, 22.0, 50.0, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 2000.0]  # except for μy as in Buckwar/Tamborrino/Tubikanec#
  #θ0 =[3.25, 100.0, 22.0, 50.0, 185.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 2000.0]  # this gives bimodality
 # θ0 =[3.25, 100.0, 22.0, 50.0, 530.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 2000.0]  # also try this one
  ℙ0 = JansenRitDiffusion(θ0...)
  @show properties(ℙ0)
  AuxType = JansenRitDiffusionAux
end
if model == :jr3
  θ0 =[3.25, 100.0, 22.0, 50.0, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 0.01, 200.0, 1.0]  # except for μy as in Buckwar/Tamborrino/Tubikanec#
  ℙ0 = JansenRitDiffusion3(θ0...)
  AuxType = JansenRitDiffusionAux3
end

#------  set observations
L = @SMatrix [0.0 1.0 -1.0 0.0 0.0 0.0]
m,  = size(L)
Σdiagel = 1e-7 # oorspr 1e-9
Σ = SMatrix{m,m}(Σdiagel*I)

#--- generate test data
T = 3.0 
x00 = @SVector [0.3, 2.0, 3.0, 2.0, 1.5, 1.0] # in generating the data, take the intial point completelly arbitrary 
W = sample((-1.0):0.0001:T, wienertype(ℙ0))                        #  sample(tt, Wiener{ℝ{1}}())
Xf_prelim = Bridge.solve(Bridge.Euler(), x00, W, ℙ0)
# drop initial nonstationary behaviour
Xf = SamplePath(Xf_prelim.tt[10001:end], Xf_prelim.yy[10001:end])
x0 = Xf.yy[1]
dt = Xf.tt[2]-Xf.tt[1]

skipobs = 400                                     # I took 400  all the time
obstimes =  Xf.tt[1:skipobs:end]
obsvals = map(x -> L*x, Xf.yy[1:skipobs:end])
pF = plot_all(ℙ0,  Xf, obstimes, obsvals)
savefig(joinpath(outdir, "forwardsimulated_truepars.png"))

prior_on_x0 = true

if !prior_on_x0      #------- process observations, assuming x0 known
  obs = [Observation(obstimes[1],  x0,  SMatrix{6,6}(1.0I), SMatrix{6,6}(Σdiagel*I))]
  for i in 2:length(obstimes)
    push!(obs, Observation(obstimes[i], obsvals[i], L, Σ));
  end
else         # -- now obs with staionary prior on x0
  zerovec = @SVector zeros(6)
  obs = [Observation(-1.0,  zerovec,  SMatrix{6,6}(1.0I), SMatrix{6,6}(Σdiagel*I))]
  for i in 1:length(obstimes)
    push!(obs, Observation(obstimes[i], obsvals[i], L, Σ));
  end
  pushfirst!(obsvals, SA[0.0])
  pushfirst!(obstimes, -1.0)
  Xf = Xf_prelim
  x0 = zerovec
end



#----------- obs and obsvals are input to mcmc algorithm
@show ℙ0


# remainder is checking 
S = DE(Vern7())

timegrids = set_timegrids(obs, 0.0005)
B = BackwardFilter(S, ℙ0, AuxType, obs, obsvals, timegrids) ;
Z = Innovations(timegrids, ℙ0);

# check
XX, ll = forwardguide(B, ℙ0)(x0, Z);
pG = plot_all(ℙ0, timegrids, XX)
l = @layout [a;b]
plot(pF, pG, layout=l)
savefig(joinpath(outdir,"forward_guidedinitial_separate_truepars.png"))

plot_all(ℙ0,Xf,  obstimes, obsvals, timegrids, XX)
savefig(joinpath(outdir,"forward_guidedinitial_overlaid_truepars.png"))


deviations = [obsvals[i] - L * XX[i-1][end]  for i in 2:length(obsvals)]
plot(obstimes[2:end], first.(deviations))
savefig(joinpath(outdir,"deviations_guidedinitial_truepars.png"))


TEST=false 
if TEST
  S = Vern7direct();
  @time BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids);


  S = DE(Vern7())
  @time  BackwardFilter(S, ℙ, AuxType, obs,obsvals, timegrids);
  # S = RK4()
  # @btime  BackwardFilter(S, ℙ, AuxType, obs, timegrids);
  @time forwardguide(x0, ℙ, Z, B);

  # using Profile
  # using ProfileView
  # Profile.init()
  # Profile.clear()
  # S = DE(Vern7())#S = Vern7direct();
  # @profile  BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids);
  # @profile parupdate!(B, XX, movetarget, obs, obsvals, S, AuxType, timegrids; verbose=verbose)(x0, ℙ, Z, ll);# θ and XX may get overwritten
  # ProfileView.view()
end