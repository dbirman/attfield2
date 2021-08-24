function fit = fitLog(msdur,dp)
%% fit an exponential to the dp data
f = @(x) x(1)*log(x(2)*msdur+1)-dp;

fit.params = lsqnonlin(f,[0.2 80]);

xs = linspace(msdur(1),msdur(end),10*length(msdur));

fit.x = xs;
fit.y = fit.params(1)*log(fit.params(2)*xs+1);