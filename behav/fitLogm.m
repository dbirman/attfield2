function fit = fitLogm(msdur,dp)
%% fit an exponential to the dp data
f = @(x) log(x*msdur)-dp;

fit.params = lsqnonlin(f,[1.5]);

xs = linspace(msdur(1),msdur(end),10*length(msdur));

fit.x = xs;
fit.y = fit.params(1)-fit.params(1)*exp(-fit.params(2)*xs);