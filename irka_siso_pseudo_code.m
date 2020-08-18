
function [V,W]=irka_siso_pseudo_code(A,E,b,c,S,tol1,tol2,max_iter)

% b = c.'; % 1-sided

% IRKA for  H(s) = c (sI-A)^{-1} b
% Assumes single-input, single-output.

% Input: A,b, c matrics and initial shift selection S
%        tol1: tolerance to determine real/comples shifts
%        tol2: tolerance for the stopping criterion
% Output: Ar,br,cr and Siter = history of the shift

% Last Modified by Gugercin on April 29, 2014.

n = size(A,2);


% Figure out the real and complex shifts
x_im = find( abs(imag(S))./abs(S) >= tol1 );
x_re = find( abs(imag(S))./abs(S) < tol1 );
Sim = S(x_im);
Sre = S(x_re);
l_Sim = length(Sim);
l_Sre = length(Sre);


% Construc V and W

Vim = []; Wim = []; Vre = []; Wre = [];
if l_Sim > 0.5
    for i=1:2:l_Sim
        Vim = [Vim (A-E*Sim(i))\b];
        Wim = [Wim (A'-E'*Sim(i)')\c']; % 1-sided
    end
    Vim = [real(Vim) imag(Vim)];
    Wim = [real(Wim) imag(Wim)]; % 1-sided
end
if length(Sre) > 0.5
    for i=1:l_Sre
        Vre = [Vre (A-E*Sre(i))\b];
        Wre = [Wre (A'-E'*Sre(i))\c']; % 1-sided
    end
else
end

[V,~]= qr(full([Vre Vim]),0);
[W,~]= qr(full([Wre Wim]),0); % 1-sided
% W = V; % 1-sided

% Reduce

Er = W'*E*V;
Ar = (W'*A*V); %br = (W'*b); cr = c*V;
Siter = sort(S);
iter_count = 1;
error_in_S = 1;
while error_in_S > tol2 && iter_count < max_iter
    
    % Reflect the Ritz values
    S_p = S;
    S = -eig(full(Ar),full(Er));
    %S  = abs(real(S)) +1i*imag(S);

    S2 = sort([S_p S]);
    error_in_S = norm(S2(:,1)-S2(:,2))/norm(S2(:,2));

    Siter = [Siter S];
    
    % Figure out the real and complex shifts
    x_im = find( abs(imag(S))./abs(S) >= tol1 );
    x_re = find( abs(imag(S))./abs(S) < tol1 );
    Sim = S(x_im);
    Sre = S(x_re);
    l_Sim = length(Sim);
    l_Sre = length(Sre);

    % COnstruc V and W
    Vim = []; Wim = []; Vre = []; Wre = [];
    if l_Sim > 0.5
        for i=1:2:l_Sim
            Vim = [Vim (A-E*Sim(i))\b];
            Wim = [Wim (A'-E'*Sim(i)')\c'];
        end

        Vim = [real(Vim) imag(Vim)];
        Wim = [real(Wim) imag(Wim)];

    end
    if length(Sre) > 0.5
        for i=1:l_Sre
            Vre = [Vre (A-E*Sre(i))\b];
            Wre = [Wre (A'-E'*Sre(i))\c'];
        end
    else
    end

    [V,~]= qr(full([Vre Vim]),0);
    [W,~]= qr(full([Wre Wim]),0);
%     W = V;

    % Reduce
    Er = W'*E*V;
    Ar = (W'*A*V); %br = (W'*b); cr = c*V;
    iter_count = iter_count+ 1;


end
