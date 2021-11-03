function fit = fitLogModel(msdur,dp_focal,dp_dist)
%% fit an exponential to the dp data for each subject

f = @(x) [x(3)*x(1)*log(x(2)*msdur+1)-dp_focal x(1)*log(x(2)*msdur+1)-dp_dist];

fit.params = lsqnonlin(f,[0.4 160 2]);

xs = linspace(msdur(1),msdur(end),10*length(msdur));

fit.x = linspace(xs(1),xs(end),1000);
fit.y_f = fit.params(3)*fit.params(1)*log(fit.params(2)*fit.x+1);
fit.y_d = fit.params(1)*log(fit.params(2)*fit.x+1);