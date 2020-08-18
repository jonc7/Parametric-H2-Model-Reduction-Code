function [V] = FindV(M,D,L,B,C,Sigma,r)
n = size(L,1);
V = sparse(n,r);

% seprating real and imaginary parts
k = 1;

while k<=r
    tempV =((Sigma(k))^2*M + Sigma(k)*D + L)\B;
    %if ~isreal(tempV)
    if ~isreal(Sigma(k))
        V(:,k:(k+1)) = [real(tempV), imag(tempV)];
        k = k+2;
    else
        V(:,k) = real(tempV);
        k = k+1;
    end
end
% make orthogonal
[V,~] = qr(V,0);
