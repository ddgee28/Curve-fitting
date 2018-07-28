function [ outStruct ] = ExponentialFit(x,y,sigma_y,tol)
%ExponentialFit: Performs a nonlinear weighted fit on data and calculates
%                uncertainties on the fit parameters [A;xi]
%                y = A*exp(-(x-x0)/xi)
%
%
%    Note: Be wary of offset exponentials. If the tail of the distribution
%          does not decay to zero, the fit will not work properly. In this
%          case it is recommended that the user either modify the code to
%          add parameters for the background (e.g. an exponential plus a
%          constant) or that the user first subtract the background noise
%          and then fit the data near the peak.
%
%    Note: This program uses Gauss-Newton iteration to determine the fit
%          parameters, which for some cases might not converge. In this
%          case, it is recommended that the tail of the distribution be
%          cut off, with most of the data kept near the peak.
%
%
%    INPUTS: x, y, (sigma_y), (tol)
%        x:       Independant variable (for an exponential distribution)
%        y:       Dependant variable (counts or intensity of x)
%        sigma_y: Uncertainties on the dependant variable
%        tol:     Relative tolerance for the numerical iteration
%
%      The inputs x, y, and possibly sigma_y must be the same size and 1xN
%      or Nx1. If sigma_y is not provided, the fit will assume equal
%      weighting and will solve for the sigma_y that makes the reduced
%      chi-squared equal to 1. If tol is not provided, the iteration will
%      use a preset tolerance for its stop criterion.
%
%
%    OUTPUTS: A structure array containing the following fields:
%        x0:           Smallest value of the independent variable
%        A:            Amplitude of the exponential
%        xi:           Decay constant of the exponential
%        sig_[var]:    Uncertainties of the fit parameters
%        corr_[vars]:  Correlation coefficients of the fit parameters
%        fit_x, fit_y: The x and y coordinates of the fit
%        chi2:         The chi-squared estimate of goodness-of-fit
%        chi2red:      Reduced chi-squared
%        sigma_y:      If not an input, estimates the uncertainty of y
%
%      For an equal-weighted fit, a sigma_y will be found such that the
%      reduced chi-squared will be 1.


x = x(:); y = y(:); %Resizes as column arrays

%Initial guess of fit parameters
x0 = min(x);
xi0 = sum(x.*y)/sum(y) - x0;
A0 = max(y);

%Shifts exponential to improve numerics
xn = (x-x0)/xi0;
yn = y/A0;
P = [1;1]; %Estimates of shifted [Amplitude; Decay Constant]

if (~exist('sigma_y','var')) %No uncertainties given (unweighted fit)
    u = 1;
    sigma_yn = ones(size(y)); %Temporarily sets all uncertainties to 1
                              %This is rescaled once sigma_y is estimated
else
    u = 0;
    sigma_y(sigma_y<=0) = min(sigma_y(sigma_y>0)); %Resets nonpositive
                                                   %uncertainties
    sigma_yn = sigma_y/A0;
end

if (~exist('tol','var')) %No relative tolerance given
    tol = 1e-12;
end

%Weight matrix
W = diag(sigma_yn.^-2);

%Exponential and Jacobian derivative with respect to fit parameters
f = @(x,x0,A,xi)(A*exp(-(x-x0)/xi));
Df = @(x,x0,A,xi)([f(x,x0,A,xi)/A,f(x,x0,A,xi).*(x-x0)/xi^2]);

%Gauss-Newton iteration
err = Inf;
while err > tol
    J = Df(xn,0,P(1),P(2));
    dyn = yn-f(xn,0,P(1),P(2));
    N = J'*W*J;
    c = J'*W*dyn;
    dP = N\c;
    P = P+dP;
    err = sqrt((c'*dP)/(dyn'*W*dyn));
end

%Unscales
A = P(1)*A0;
xi = P(2)*xi0;

df = length(x)-length(P); %Degrees of freedom
dy = y-f(x,x0,A,xi);
J = Df(x,x0,A,xi);
N = J'*W*J;

if u==1 %Unweighted fit; W (and hence N) is scaled to match uncertainties
    chi2red = 1;
    chi2 = df;
    s_y = sqrt((dy'*dy)/df); %Estimates the uncertainty in y
    V = N\eye(size(N))*s_y^2; %Matrix of variances and covariances
else %Weighted fit; W (and hence N) is unscaled here
    chi2 = dy'*W*dy/A0^2;
    chi2red = chi2/df;
    V = N\eye(size(N))*A0^2; %Matrix of variances and covariances
end

outStruct = struct();
outStruct.x0 = x0;
outStruct.A = A;
outStruct.sig_A = sqrt(V(1,1));
outStruct.xi = xi;
outStruct.sig_xi = sqrt(V(2,2));
outStruct.corr_A_xi = V(1,2)/sqrt(V(1,1)*V(2,2));

outStruct.fit_x = x;
outStruct.fit_y = f(x,x0,A,xi);
outStruct.chi2 = chi2;
outStruct.chi2red = chi2red;
if u==1
    outStruct.sigma_y = s_y;
end
end %Fine structure at high energies