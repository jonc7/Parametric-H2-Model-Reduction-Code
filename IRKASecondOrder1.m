function [V,Mr,Dr,Lr,Br,Cr,Ar,Er,Sigma,Sigma_Change,Iter]=IRKASecondOrder1(M,D,L,B,C,Sigma,maxiter,contol,r)
%--------------------------------------------------------------------------
% One sided secondOrder IRKA (V = W)
%--------------------------------------------------------------------------
if exist('OCTAVE_VERSION', 'builtin')
    cplxpair_tol = 1000.0*eps; % increased tolerance for Octave
else
    cplxpair_tol = 100.0*eps; % the default value used in Matlab.
end

n = size(L,1);


Sigma = cplxpair(Sigma, cplxpair_tol);

% Find Bases
[V] = FindV(M,D,L,B,C,Sigma,r);

Mr = full(V'*M*V);
Dr = full(V'*D*V);
Lr = full(V'*L*V);
Br = full(V'*B);
Cr = full(C*V);

%-----------------------------
% transform to the first order state space
n1 = size(Lr,1);
A2r = [sparse(n1,n1) speye(n1);...
    -Lr -Dr];
E2r  = [speye(n1) sparse(n1,n1);...
    sparse(n1,n1)    Mr];
B2r = [sparse(n1,1) ;Br];
C2r = [Cr sparse(1,n1)];
%-----------------------------------
%  initalize standard parameters
sigma_change = inf;
iter=1;
% fprintf('Second Order IRKA iteration: %3d\n',iter);
%-------------------------------------------------------------------------
% reduce the dimension from 2r to r using BT / IRKA
% This is the strategy to choose r among 2r interpolation points

try
    [Ar,~,~,Er] = BT_Red(A2r,B2r,C2r,E2r,r);
catch
    [Vt,~] = irka_siso_pseudo_code(A2r,E2r,B2r,C2r,Sigma,1e-5,1e-5,100);
    Ar = Vt'*A2r*Vt; Er = Vt'*E2r*Vt;
end
%--------------------------------------------------------------------------

[eigenvecs, eigenvals] = eig(full(Ar),full(Er));  % calculate the eigenvalues
Sigma  =  cplxpair(-diag(eigenvals), cplxpair_tol);
% b = (eigenvecs\(E_red\B_red)).';
% c = (C_red*eigenvecs).';

Iter = [];
Sigma_Change = [];

while (sigma_change > contol && iter <= maxiter)
    [V] = FindV(M,D,L,B,C,Sigma,r);
    
    sigma_old = Sigma;
    
    Mr = full(V'*M*V);
    Dr = full(V'*D*V);
    Lr = full(V'*L*V);
    Br = full(V'*B);
    Cr = full(C*V);
    %--------------------------------
    % transform to the first order state space
    n1 = size(Lr,1);
    A2r = [sparse(n1,n1) speye(n1);...
        -Lr -Dr];
    E2r  = [speye(n1) sparse(n1,n1);...
        sparse(n1,n1)    Mr];
    B2r = [sparse(n1,1) ;Br];
    C2r = [Cr sparse(1,n1)];
    %----------------------------------------------------------------------
    % reduce the dimension from 2r to r using BT / IRKA
    % This is the strategy to choose r among 2r interpolation points
    try
        [Ar,~,~,Er] = BT_Red(A2r,B2r,C2r,E2r,r);
    catch
        [Vt,~] = irka_siso_pseudo_code(A2r,E2r,B2r,C2r,Sigma,1e-5,1e-5,100);
        Ar = Vt'*A2r*Vt; Er = Vt'*E2r*Vt;
    end

    %----------------------------------------------------------------------
    % Compute new shifts and tangential directions and continue algorithm.
    [eigenvecs, eigenvals] = eig(full(Ar),full(Er));  % calculate the eigenvalues
    Sigma  =  cplxpair(-diag(eigenvals), cplxpair_tol); %pair conjugates
    %     % update directions:
    %     b = (eigenvecs\(E_red\B_red)).';
    %     c = (C_red*eigenvecs).';
    
    % Compute the relative change for the stopping criterion
    sigma_change = norm(Sigma - sigma_old)/norm(sigma_old); % difference
    iter = iter+1;
%     fprintf('Second Order IRKA iteration: %3d\n',iter);
    
    Iter = [Iter iter];
    Sigma_Change = [Sigma_Change sigma_change];
    
end
Br = full(V'*B);
Cr = full(C*V);
