function [v,a,Ter] = ezdiffusion( Pc, vrt, mrt )
% INPUTS
%   Pc = % correct
%   vrt = variance of rt
%   mrt = mean of rt
% OUTPUTS
%   v = drift rate
%   a = boundary distance
%   Ter = non-decision time
%
s = 0.1;
s2 = s^2;

if Pc==0 || Pc==0.5 || Pc==1
    error('Pc can''t be 0/0.5/1');
end

% get the logit
L = log(Pc / (1-Pc));
x = L*(L*Pc^2 - L*Pc + Pc - 0.5)/vrt;
v = sign(Pc-0.5)*s*x^0.25;
a = s2*L/v;
y = -v*a/s2;
mdt = (a/(2*v)) * (1-exp(y))/(1+exp(y));
Ter = mrt-mdt;
