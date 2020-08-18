function [Ar,Br,Cr,Er] = BT_Red(A,B,C,E,r)
if exist('OCTAVE_VERSION', 'builtin')
    cplxpair_tol = 1000.0*eps; % increased tolerance for Octave
else
    cplxpair_tol = 100.0*eps; % the default value used in Matlab.
end
%----------------------
% Balanced Truncation
%----------------------

% compute cholesky factor of P
U = lyapchol(A,B,E);
U = U';
% compute cholesky factor of Q
L = lyapchol(A',C',E');
L = L';

[Z,S,Y] = svd(U'*E'*L);

Z1 = Z(:,1:r);
Y1 = Y(:,1:r);
S1 = S(1:r,1:r);

W = L*Y1*inv(sqrt(S1));
V = U*Z1*inv(sqrt(S1));

Er = W'*E*V;
Ar = W'*A*V;
Br = W'*B;
Cr = C*V;
%--------------------------------------------------------------------------
% % diagonalization
% [T,lambda] = eig(full(A),full(E));
% N = length(lambda);
% lambda = cplxpair(diag(lambda), cplxpair_tol);
% % for complex and real eigen values(considering both factorization and
% % diagonalization processes!)
% i = 1;
% while i<=N
%     if ~isreal(lambda(i))
%         T(:,i:(i+1)) = [real(T(:,i)), imag(T(:,i))];
%         i = i+2;
%     else
%         T(:,i) = real(T(:,i));
%         i = i+1;
%     end
% end
% %------------------
% A_tild = A*T;
% E_tild = E*T;
% B_tild = B;
% C_tild = C*T;
% 
% eig(A_tild,E_tild)
% % parts related to the zero eigenvalue
% A2 = A_tild(61,61);
% B2 = B_tild(61,:);
% C2 = C_tild(:,61);
% 
% 
% % reduced sys without zero eigen value
% A_tild(61,:) = []; % removing one row
% A_tild(:,61) = []; % removing one column
% A1 = A_tild;
% 
% B_tild(61,:) = []; % removing one row
% B1 = B_tild;
% 
% C_tild(:,61) = []; % removing one column
% C1 = C_tild;
% 
% E1  = speye(size(A1,1));
% 
% %--------------------------------------------------------------------------
% % reduction demension
% r = 10;
% %------------------------
% % %% ------------------------------------------------------------------------`
% % %  Balanced truncation (version of square root implementation)
% % %--------------------------------------------------------------------------
% % compute cholesky factor of P
% U_p = lyapchol(A1,B1);
% U_p = U_p';       %generate a lower triangle matrix
% % compute cholesky factor of Q
% L = lyapchol(A1',C1');
% L = L';
% [Z,S,Y] = svd(U_p'*L);
% 
% Z1 = Z(:,1:r);
% Y1 = Y(:,1:r);
% S1 = S(1:r,1:r);
% 
% W = L*Y1*inv(sqrt(S1));
% V = U_p*Z1*inv(sqrt(S1));
% 
% Er = W'*V;
% Ar = W'*A1*V;
% Br = W'*B1;
% Cr = C1*V;