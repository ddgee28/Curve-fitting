function [ outStruct ] = LinearFit(x,y,sigma_y)
%LinearFit: Performs a linear weighted fit on data and calculates
%           uncertainties on the fit parameters [m;b]
%           y = m*x+b
%
%
%    INPUTS: x, y, (sigma_y)
%        x:       Independant variable
%        y:       Dependant variable
%        sigma_y: Uncertainties on the dependant variable
%
%      The inputs x, y, and possibly sigma_y must be the same size and 1xN
%      or Nx1. If sigma_y is not provided, the fit will assume equal
%      weighting and will solve for the sigma_y that makes the reduced
%      chi-squared equal to 1.
%
%
%    OUTPUTS: A structure array containing the following fields:
%        m:            Slope of the line
%        b:            Intercept of the line
%        sig_[var]:    Uncertainties of the fit parameters
%        corr_[vars]:  Correlation coefficients of the fit parameters
%        fit_x, fit_y: The x and y coordinates of the fit
%        chi2:         The chi-squared estimate of goodness-of-fit
%        chi2red:      Reduced chi-squared
%        sigma_y:      If not an input, estimates the uncertainty of y
%
%      For an equal-weighted fit, a sigma_y will be found such that the
%      reduced chi-squared will be 1.

n = length(x);
x = x(:); y = y(:);
if (~exist('sigma_y','var')) %Unweighted fit
    J = [ones(size(x)),x];
    N = J'*J;
    P = N\(J'*y);
    b=P(1); m=P(2);
    dy = y-m*x-b;
    chi2red = 1;
    chi2 = n-length(P);
    s_y = sqrt((dy'*dy)/(n-length(P)));
    V = N\eye(size(N))*s_y^2;
else %Weighted fit
    sigma_y = sigma_y(:);
    W = diag(1./(sigma_y.^2));
    J = [ones(size(x)),x];
    N = J'*W*J;
    P = N\(J'*W*y);
    b=P(1); m=P(2);
    dy = y-m*x-b;
    chi2 = dy'*W*dy;
    chi2red = chi2/(n-length(P));
    V = N\eye(size(N));
end

outStruct = struct();
outStruct.m = m;
outStruct.sig_m = sqrt(V(2,2));
outStruct.b = b;
outStruct.sig_b = sqrt(V(1,1));
outStruct.corr_m_b = V(1,2)/sqrt(V(1,1)*V(2,2));
outStruct.fit_x = x;
outStruct.fit_y = m*x+b;
outStruct.chi2 = chi2;
outStruct.chi2red = chi2red;
if exist('s_y','var')
    outStruct.sigma_y = s_y;
end
end