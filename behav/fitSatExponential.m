function fit = fitSatExponential(msdur,dp)
%% fit an exponential to the dp data
f = @(x) (x(1)-x(1)*exp(-x(2)*msdur))-dp;

fit.params = lsqnonlin(f,[1.5 25]);

xs = linspace(msdur(1),msdur(end),10*length(msdur));

fit.x = xs;
fit.y = fit.params(1)-fit.params(1)*exp(-fit.params(2)*xs);