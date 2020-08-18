% Jon Cooper Research GP Surrogate for POD Iterative Algorithm
% addpath(genpath(pwd))
% clear all;
close all;
rng(0);
set(0, 'DefaultLineLineWidth', 2);

%% Setup
MODEL = 4;           % 1 - Convection-Diffusion, 2 - Beam, 3 - Artificial, 4 - Modified Gyro
FORDER = 500;        % full system order
% u = @(t) exp(-t)*sin(t);%+sin(t/2); % input to the dynamical system

ModelTol = 1e-2;     % maximum desired error (inf norm in the p-space)
Ni = 50;             % number of samples on the imaginary axis to integrate
trueError = 0;       % boolean to compute & plot the true errors
Est1 = 1;            % boolean to sample from estimate 1
Var1 = 1;            % boolean to sample from variant 1
Var4 = 1;            % boolean to sample from variant 4

% FIXED frequency samples for every parameter sample (might make adaptive later)
sSamp = 1i*logspace(-6,6,2).'; sSamp = [sSamp;-sSamp];
% sSamp2 = sSamp-1i; % need different samples for dual and residual systems
sSamp2 = 1i*logspace(-5,6,2).';
sSamp2 = [sSamp2;-sSamp2];


%% Full Order Model Construction
if MODEL == 1 && mod(FORDER^.5,1) ~= 0 % check/correction is specific to model
    FORDER = ceil(FORDER^.5)^2;
    warning('Setting full order to %d',FORDER);
elseif MODEL == 2 && mod(FORDER,4) ~= 0
    FORDER = ceil(FORDER/4)*4;
    warning('Setting full order to %d',FORDER);
elseif MODEL == 4 && FORDER ~= 17931
    FORDER = 17931;
    warning('Setting full order to 17931');
end

switch(MODEL)
    case 1 % Convection-Diffusion Flow Model
        second = 0;
        fprintf('Model: Convection-Diffusion Flow\n');
        n = FORDER^.5;
        [A0, A1, A2] = fdm_2d_matrix_2p(n,'1','1','0');
        p1 = 0.1;  p0 = 0.5; % PARAMETER SETTINGS (p0 -> K)
        Ac = p1*A1 + p0*A0; Ap = A2;
        A = @(p) Ac + p*Ap;
        B = sparse(ones(FORDER,1));
        C = sparse(ones(1,FORDER));
        E = speye(FORDER);
        FullSS = @(p) dss(full(A(p)),full(B),full(C),0,full(E));
        Q = @(s,p) s*E-A(p);
        
        pbnd = [-3,3]; % !LOG! bounds ([lower,upper]) for p
        
    case 2 % Beam Model
        second = 1;
        fprintf('Model: Beam\n');
        n = FORDER/4;
        n2  = 2*n;
        [M,K]=finbeam(80,n,2700,.00651*.02547,.00651^3*.02547/12,7.1e10);
        M  = sparse(M); K  = sparse(K);
%         B2 = eye(n2,1); C2 = eye(n2,1)';
        a = 0.1; % fixed parameters
        D = @(p) p*K+a*M;
%         H = @(s) C2*((s^2*M + s*(a*M+b*K) + K)\B2);
        
        % First-order form
        Z = sparse(n2,n2);  I = speye(n2);
        A0 = [Z I ; -K Z];
        A1 = [Z Z ; Z -M];
        A2 = [Z Z ; Z -K];
        Ac = A0 + a*A1; Ap = A2;
        A = @(p) Ac + p*Ap;
        
        B2 = eye(n2,1); B = [zeros(n2,1); B2];
        C2 = eye(FORDER/2,1)'; C = eye(FORDER,1)';
        E = [I Z ; Z M];
        
        FullSS = @(p) dss(full(A(p)),full(B),full(C),0,full(E));
        
        B = B2; C = C2;
        Q = @(s,p) s^2*M + s*(a*M+p*K) + K;
        
        pbnd = [-3,log10(1/a)]; % !LOG! bounds ([lower,upper]) for p
        
    case 3 % artificial model
        second = 0;
        Ac = diag(-10*(1:FORDER));
        temp = orth(rand(FORDER));
        Ac = temp*Ac*temp.'; Ac = .5*(Ac+Ac.');
%         Ap = diag(-abs(10*rand(FORDER,1)));
        Ap = diag(-.1*(1:FORDER));
        temp = rand(FORDER);
        Ap = temp*Ap/temp; %Ap = .5*(Ap+Ap.');
        
        A = @(p) Ac + p*Ap;
        B = rand(FORDER,1);
        C = rand(1,FORDER);
%         C = B.';
        E = eye(FORDER);
        
        FullSS = @(p) dss(A(p),B,C,0,E);
        Q = @(s,p) s*E-A(p);
        
        pbnd = [-3,3]; % !LOG! bounds ([lower,upper]) for p
        
    case 4 % butterfly gyroscope
        second = 1;
        fprintf('Model: Butterfly Gyroscope\n');
        % matrices
        B = sparse(mmread('gyroscope.B'));
        D1 = mmread('gyroscope.D1');
        D2 = mmread('gyroscope.D2');
        M1 = mmread('gyroscope.M1');
        M2 = mmread('gyroscope.M2');
        T1 = mmread('gyroscope.T1');
        T2 = mmread('gyroscope.T2');
        T3 = mmread('gyroscope.T3');
        C = speye(1,FORDER); C(1) = 0; C(2315) = -1; C(5806) = 1;
        % parameters
        theta = [1e-7,1e-5]; theta = 1e-6;
        alpha = 0.1;
        beta = 1e-9;
        d = [1,2]; %d = 1.5;
        M = @(d) M1+d*M2; 
        K = @(d) T1+T2/d+d*T3;
        D = @(d) theta*(D1+d*D2)+alpha*M(d)+beta*K(d);
        
        Q = @(s,p) s^2*M1+s^2*p*M2 + s*theta*D1+s*theta*p*D2+...
            +s*alpha*M1+s*alpha*p*M2+s*beta*T1+s*beta*T2/p+s*beta*p*T3 +...
            +T1+T2/p+p*T3;
        
        pbnd = [0,log10(2)]; % !LOG! bounds ([lower,upper]) for p
        FullSS = @(p) 0;
        if trueError
            warning('Disabling true error plots due to size.');
            trueError = 0;
        end
        
    case 5 % thermal model
        
        
end
%%
pHist = 10^(pbnd(1)+(pbnd(2)-pbnd(1))*rand(1)); % must be a row vector, initial parameter sample
p = pHist; orderHist = [];

V = []; W = [];
Vdu = []; Wdu = [];
Vrdu = []; Wrdu = [];
Vrpr = []; Wrpr = [];

iter = 1;
while 1
    fprintf("Iteration: %d\n",iter);
    s = sSamp; s2 = sSamp2;
    np = length(pHist) - nnz(abs(pHist-p) > 1e-10); % number of times p appears
    if np > 1
        np = np-1;
%         s = [s;1i*rand(np*length(sSamp),1)];
%         s2 = [s2;1i*rand(np*length(sSamp2),1)];
        s = [s(1:length(s)/2);1i*rand(np*length(sSamp)/2,1)];  s = [s;-s];
        s2 = [s2(1:length(s2)/2);1i*rand(np*length(sSamp2)/2,1)];  s2 = [s2;-s2];
    end
    %% ROMs
    fprintf("Constructing ROMs... ");
    % ROM for Primal System
    if second
        [Vn,~,~,~,~,~,~,~,sigma,sigma_change,Iter] = IRKASecondOrder1(M(p),D(p),K(p),B,C,s,10,5e-3,length(s));
        Vn = full(Vn); V = orth([V,Vn]); W = V;
        M1r = V'*M1*V; M2r = V'*M2*V; D1r = V'*D1*V; D2r = V'*D2*V;
        T1r = V'*T1*V; T2r = V'*T2*V; T3r = V'*T3*V;
        Mr = @(d) M1r+d*M2r; Kr = @(d) T1r+T2r/d+d*T3r;
        Dr = @(d) theta*(D1r+d*D2r)+alpha*Mr(d)+beta*Kr(d); Bp = V'*B; Cp = C*V;
%         MpKr = V'*MpK*V; Dfr = V'*Df*V; % GYRO
        n1 = size(V,2);
%         A2rp = @(p) full([sparse(n1,n1) speye(n1);...
%             -Kr -(p*Kr+a*Mr)]);
        A2rp = @(p) full([sparse(n1,n1) speye(n1);... % GYRO
            -Kr(p) -Dr(p)]);
        E2p  = @(p) full([speye(n1) sparse(n1,n1);...
            sparse(n1,n1)    Mr(p)]);
        B2p = full([sparse(n1,1) ;Bp]);
        C2p = full([Cp sparse(1,n1)]);
        RedSS = @(p) dss(A2rp(p),B2p,C2p,0,E2p(p));
%         Qh = @(s,p) s^2*Mr + s*(a*Mr+p*Kr) + Kr;
        Qh = @(s,p) s^2*Mr(p) + s*Dr(p) + Kr(p); % GYRO
    else
        [Vn,~] = irka_siso_pseudo_code(A(p),E,B,C,s,1e-5,1e-5,100);
        V = orth([V,Vn]); W = V;
        Acp = W'*Ac*V; App = W'*Ap*V; % primal reduced matrices
        Arp = @(p) Acp+p*App; Ep = W'*E*V;
        Bp = W'*B; Cp = C*V;
        RedSS = @(p) dss(Arp(p),Bp,Cp,0,Ep);
        Qh = @(s,p) s*Ep-Arp(p);
    end

    xprh = @(s,p) V*(Qh(s,p)\Bp); % solution to reduced order system
    rpr = @(s,p,Q) B - Q*xprh(s,p); % residual of reduced order system

    % ROM for Dual System
    if second
        [Vdun,~,~,~,~,~,~,~,~,~,~] = IRKASecondOrder1(M(p).',D(p).',K(p).',C.',B.',s,10,5e-3,length(s));
        Vdun = full(Vdun); Vdu = orth([Vdu,Vdun]); Wdu = Vdu;
        M1dur = Vdu'*M1*Vdu; M2dur = Vdu'*M2*Vdu; D1dur = Vdu'*D1*Vdu; D2dur = Vdu'*D2*Vdu;
        T1dur = Vdu'*T1*Vdu; T2dur = Vdu'*T2*Vdu; T3dur = Vdu'*T3*Vdu;
        Mdur = @(d) M1dur+d*M2dur; Kdur = @(d) T1dur+T2dur/d+d*T3dur;
        Ddur = @(d) theta*(D1dur+d*D2dur)+alpha*Mdur(d)+beta*Kdur(d); Crdu = (C*Vdu)';
%         Qduh = @(s,p) s^2*Mdur + s*(a*Mdur+p*Kdur) + Kdur;
%         MpKdur = Vdu'*MpK*Vdu; Dfdur = Vdu'*Df*Vdu;
        Qduh = @(s,p) s^2*Mdur(p) + s*Ddur(p) + Kdur(p);
    else
        [Vdun,~] = irka_siso_pseudo_code(A(p).',E.',C.',B.',s,1e-5,1e-5,100);
        Vdu = orth([Vdu,Vdun]); Wdu = Vdu;
        Acd = (Wdu'*Ac*Vdu).'; Apd = (Wdu'*Ap*Vdu).'; % dual reduced matrices
        Ard = @(p) Acd+p*Apd; Ed = (Wdu'*E*Vdu).';
        Crdu = Wdu'*C';
        Qduh = @(s,p) s*Ed-Ard(p);
    end
    
    xduh = @(s,p) Vdu*(Qduh(s,p)\Crdu); % solution to reduced dual system
    rdu = @(s,p,Q) C.' - Q.'*xduh(s,p); % residual

    % ROM for Dual-Residual System
    if second
        [Vrdun,~,~,~,~,~,~,~,~,~,~] = IRKASecondOrder1(M(p).',D(p).',K(p).',C.',B.',s,1,1e-5,length(s));
        Vrdun = full(Vrdun); Vrdu = orth([Vrdu,Vrdun]); Vrdu = orth([Vrdu,Vdu]); Wrdu = Vrdu;
        M1rdu = Vrdu'*M1*Vrdu; M2rdu = Vrdu'*M2*Vrdu; D1rdu = Vrdu'*D1*Vrdu; D2rdu = Vrdu'*D2*Vrdu;
        T1rdu = Vrdu'*T1*Vrdu; T2rdu = Vrdu'*T2*Vrdu; T3rdu = Vrdu'*T3*Vrdu;
        Mrdu = @(d) M1rdu+d*M2rdu; Krdu = @(d) T1rdu+T2rdu/d+d*T3rdu;
        Drdu = @(d) theta*(D1rdu+d*D2rdu)+alpha*Mrdu(d)+beta*Krdu(d);
%         Qrdu = @(s,p) s^2*Mrdu + s*(a*Mrdu+p*Krdu) + Krdu;
%         MpKrdu = Vrdu'*MpK*Vrdu; Dfrdu = Vrdu'*Df*Vrdu;
        Qrdu = @(s,p) s^2*Mrdu(p) + s*Drdu(p) + Krdu(p);
    else
        [Vrdun,~] = irka_siso_pseudo_code(A(p).',E.',C.',B.',s2,1e-5,1e-5,1);
        Vrdu = orth([Vrdu,Vrdun]); Vrdu = orth([Vrdu,Vdu]); Wrdu = Vrdu;
        Acdr = (Wrdu'*Ac*Vrdu).'; Apdr = (Wrdu'*Ap*Vrdu).'; % dual residual reduced matrices
        Ardr = @(p) Acdr+p*Apdr; Edr = (Wrdu'*E*Vrdu).';
        Qrdu = @(s,p) s*Edr-Ardr(p);
    end
    
    xrduh = @(s,p,Q) Vrdu*(Qrdu(s,p)\(Wrdu.'*rdu(s,p,Q))); % reduced solution

    % ROM for Primal-Residual System
    if second
        [Vrprn,~,~,~,~,~,~,~,~,~,~] = IRKASecondOrder1(M(p),D(p),K(p),B,C,s,1,1e-5,length(s));
        Vrprn = full(Vrprn); Vrpr = orth([Vrpr,Vrprn]); Vrpr = orth([Vrpr,V]); Wrpr = Vrpr;
        M1rpr = Vrpr'*M1*Vrpr; M2rpr = Vrpr'*M2*Vrpr; D1rpr = Vrpr'*D1*Vrpr; D2rpr = Vrpr'*D2*Vrpr;
        T1rpr = Vrpr'*T1*Vrpr; T2rpr = Vrpr'*T2*Vrpr; T3rpr = Vrpr'*T3*Vrpr;
        Mrpr = @(d) M1rpr+d*M2rpr; Krpr = @(d) T1rpr+T2rpr/d+d*T3rpr;
        Drpr = @(d) theta*(D1rpr+d*D2rpr)+alpha*Mrpr(d)+beta*Krpr(d);
%         Qrpr = @(s,p) s^2*Mrpr + s*(a*Mrpr+p*Krpr) + Krpr;
%         MpKrpr = Vrpr'*MpK*Vrpr; Dfrpr = Vrpr'*Df*Vrpr;
        Qrpr = @(s,p) s^2*Mrpr(p) + s*Drpr(p) + Krpr(p);
    else
        [Vrprn,~] = irka_siso_pseudo_code(A(p),E,B,C,s2,1e-5,1e-5,1);
        Vrpr = orth([Vrpr,Vrprn]); Vrpr = orth([Vrpr,V]); Wrpr = Vrpr;
        Acpr = Wrpr'*Ac*Vrpr; Appr = Wrpr'*Ap*Vrpr; % primal residual reduced matrices
        Arpr = @(p) Acpr+p*Appr; Epr = Wrpr'*E*Vrpr;
        Qrpr = @(s,p) s*Epr-Arpr(p);
    end
    
    xrprh = @(s,p,Q) Vrpr*(Qrpr(s,p)\(Wrpr.'*rpr(s,p,Q))); % reduced solution
    
    %% Error Estimates
    delta1 = @(s,p,Q) abs(xduh(s,p).'*rpr(s,p,Q)); % Error Estimate 1 (prop. 3.1)
    delta2 = @(s,p,Q) abs(xrduh(s,p,Q).'*rpr(s,p,Q));
    delta2pr = @(s,p,Q) abs(rdu(s,p,Q).'*xrprh(s,p,Q)); % Error Estimate Variant 1 (thm 4.1)
    delta1pr = @(s,p,Q) abs(C*xrprh(s,p,Q)); % Error Estimate Variant 2 (thm 4.2)
    rrpr = @(s,p,Q) rpr(s,p,Q)-Q*xrprh(s,p,Q); % Error Estimate Variant 3a (thm 4.3) (just an upper bound)
    delta3 = @(s,p,Q) abs(xduh(s,p).'*rrpr(s,p,Q)); % Error Estimate Variant 3b (thm 4.4)
    
    fprintf("Done.\n");
    fprintf("Orders: %d, %d, %d, %d\n",size(V,2),size(Vdu,2),size(Vrpr,2),size(Vrdu,2))
    orderHist = [orderHist,size(V,2)];
    %% Optimization
    fprintf("Optimizing... ");
    errorApx = @(p) samplerTrap(p,Est1,Var1,Var4,Q,delta1,delta2,delta2pr,delta1pr,delta3,RedSS,Ni);
    [p,error] = GridSrch(errorApx,pbnd,@(p) norm(FullSS(p)-RedSS(p),2),pHist,FullSS,trueError); %error = -error;
    fprintf("Done.\n");
    fprintf("p: %e, error: %e\n",p,error);
    %% Exit Conditions
    if error <= ModelTol
        break
    end
    if ~all(pHist-p)
%         warning('Resampled at p = %f\n',p);
        pHist = [pHist,p];
%         break
    else
        pHist = [pHist,p];
    end
    iter = iter+1;
end

%% testing timing

N = 10;
pp = logspace(pbnd(1),pbnd(2),N);

if trueError
    tic
    for i = 1:N % exact error
        norm(FullSS(pp(i))-RedSS(pp(i)),2);
    end
    fprintf('True Error Time:\n');
    toc
end
if Est1
    tic
    for i = 1:N % Estimate 1
        samplerTrap(pp(i),1,0,0,Q,delta1,delta2,delta2pr,delta1pr,delta3,RedSS,Ni);
    end
    fprintf('Estimate 1 Time:\n');
    toc
end
if Var1
    tic
    for i = 1:N % Variant 1
        samplerTrap(pp(i),0,1,0,Q,delta1,delta2,delta2pr,delta1pr,delta3,RedSS,Ni);
    end
    fprintf('Variant 1 Time:\n');
    toc
end
if Var4
    tic
    for i = 1:N % Variant 4
        samplerTrap(pp(i),0,0,1,Q,delta1,delta2,delta2pr,delta1pr,delta3,RedSS,Ni);
    end
    fprintf('Variant 4 Time:\n');
    toc
end

%% Plot Details

n = 7; if trueError, n = 8; end
fend = get(gcf,'Number');
switch MODEL
    case 1, ti = 'CV-Flow';
    case 2, ti = 'Beam';
    case 3, ti = 'Artificial';
    case 4, ti = 'Gyroscope';
end
for i = 1:fend
    figure(i);
    yli = ylim; yl = yli(1); yu = yli(2);
    title([ti,' | Full Order ',num2str(FORDER),' | Step ',num2str(i),' | Reduced Order ',num2str(orderHist(i))]);
    if trueError
        legend('Variance','Mean Approx.','True','Approx. 1','Approx. 2','Approx. 3');
    else
        legend('Variance','Mean Approx.','Approx. 1','Approx. 2','Approx. 3');
    end
%     set(gca, 'YScale', 'log'); % set log Y axis
%     yli = ylim; yu = yli(2); % keep upper Y lim
%     ylim([0,yu]);
    fig = gcf; dataObjs = findobj(fig,'-property','YData');
    std = dataObjs(n).YData; std(std < 0) = yl; dataObjs(n).YData = std;
    xlim(10.^pbnd);
    xlabel('p'); ylabel('Approximate Relative $\mathcal{H}_2$ Error','interpreter','LaTeX');
end

%% GP Fit
% ii = 1:10:length(pp); % indices to fit
% trainY = [lowerest1(ii)',upperest1(ii)',lowerest2(ii)',upperest2(ii)',lowerest3(ii)',upperest3(ii)'];
% trainX = log(repmat(pp(ii),1,6));
% [mu,Sigma] = gaussianProcess(trainX',trainY',log(pp)');
% 
% % plotGP
% p = [0.025, 0.975]; % percentiles
% Sigma = sqrt(diag(Sigma));
% ql = norminv(p(1),mu,Sigma);
% qu = norminv(p(2),mu,Sigma);

% figure; hold on;
% % color = 0.8* [1 1 1];
% % xx = [pp(:);flipud(pp(:))];
% % yy =  [qu(:); flipud(ql(:))];
% % h  = fill(xx,yy,color,'EdgeColor', color); hold on;
% % plot(pp,mu,'LineWidth',2); hold on;
% % set(h,'facealpha',.5);
% 
% % plot(pp(ii),mu(ii),'xr','MarkerSize',15);
% if trueError
%     plot(pp,H2true,'-k');
% end
% if Est1
%     loglog(pp,lowerest1,'--r');
%     loglog(pp,upperest1,'--r','HandleVisibility','off');
% end
% if Var1
%     loglog(pp,lowerest2,'--g');
%     loglog(pp,upperest2,'--g','HandleVisibility','off');
% end
% if Var4
%     loglog(pp,lowerest3,'--b');
%     loglog(pp,upperest3,'--b','HandleVisibility','off');
% end
% set(gca, 'XScale', 'log')


%% Sampling Function
function out = samplerTrap(p,Est1,Var1,Var4,Q,d1,d2,d2pr,d1pr,d3,RedSS,N)
    out = zeros(N,2*((Est1 ~= 0)+(Var1 ~= 0)+(Var4 ~= 0)));
    xx = logspace(-6,6,N);
    xxi = 1i*xx;
    for j = 1:N
        s = xxi(j);
        Qj = Q(s,p); % only 1 eval of Q per freq sample s
        ct = 1;
        if Est1
            out(j,ct) = d1(s,p,Qj);
            out(j,ct+1) = out(j,ct)+d2(s,p,Qj);
            ct = ct + 2;
            if Est1, out(j,ct) = out(j,ct-2); end
        end
        if Var1
            if ~Est1, out(j,ct) = d1(s,p,Qj); end
            out(j,ct+1) = out(j,ct)+d2pr(s,p,Qj);
            ct = ct + 2;
        end
        if Var4
            out(j,ct) = d1pr(s,p,Qj);
            out(j,ct+1) = out(j,ct)+d3(s,p,Qj);
        end
    end
    out = (trapz(xx,out.^2)/pi).^.5;
    out = out.'./norm(RedSS(p));
end

function out = certainty(fcn,p)
    out = fcn(p);
    out = mean(out)+diag(sqrt(diag(var(out+1e-8))))';
end

%% Grid Search
function [p,error] = GridSrch(fcn,bnd,tfcn,pHist,RedSS,trueError)
    N = 10;
    pp = logspace(bnd(1),bnd(2),N)';
    out = arrayfun(fcn,pp,'UniformOutput',false);
    nout = length(out{1});
    out = reshape(cell2mat(out),N*nout,1);
    mu = arrayfun(@(i) mean(out(i:i+nout-1)),1:nout:length(out)-nout+1);
    Sigma = arrayfun(@(i) var(out(i:i+nout-1)),1:nout:length(out)-nout+1);
    Sigma = sqrt(diag(Sigma+1e-8));
    
    [~,p] = max(mu+diag(Sigma)');
    objfcn = @(p) -certainty(fcn,p);
    if p == 1
        lower = pp(1);
        upper = pp(2);
    elseif p == N
        lower = pp(N-1);
        upper = pp(N);
    else
        lower = pp(p-1);
        upper = pp(p+1);
    end
    opts = optimset('MaxFunEval',10,'Display','off');
    [p,error] = fminbnd(objfcn,lower,upper,opts); error = -error;
    
    
    %% plot
    if trueError
        H2true = cell2mat(arrayfun(tfcn,pp,'UniformOutput',false));
        H2true = H2true./arrayfun(@(p) norm(RedSS(p)),pp);
    end
    
    pc = [0.025, 0.975]; % percentiles
    ql = diag(norminv(pc(1),mu,Sigma));
    qu = diag(norminv(pc(2),mu,Sigma));
    figure;
    color = 0.8* [1 1 1];
    xx = [pp;flipud(pp)];
    yy =  [qu(:); flipud(ql(:))];
    h  = fill(xx,yy,color,'EdgeColor', color); hold on;
    plot(pp,mu,'LineWidth',2);
    if trueError, plot(pp,H2true,'-k'); end
%     plot(pp,out(1:nout:end),'--m');
%     plot(pp,out(2:nout:end),'--m','HandleVisibility','off');
    plot(pp,out(1:nout:end),'--b');
    plot(pp,out(2:nout:end),'--b','HandleVisibility','off');
    plot(pp,out(3:nout:end),'--g');
    plot(pp,out(4:nout:end),'--g','HandleVisibility','off');
    plot(pp,out(5:nout:end),'--m');
    plot(pp,out(6:nout:end),'--m','HandleVisibility','off');
    set(h,'facealpha',.5);
    set(gca, 'XScale', 'log')
    set(gca, 'YScale', 'log')
    yl = get(gca,'ylim'); yl = yl(1);
    plot(pHist,yl*ones(size(pHist)),'+b');
    plot([p,p],[yl,error],'+r');
end

%% Expected Improvement
% note: assumes p should be on a log-scale
function [p,error] = EI(fcn,bnd,niter)
    p0 = bnd(1)+(bnd(2)-bnd(1))*rand(2,1); % random number in the bounded range
    pp = p0(1); p = p0(2);
    trainY = fcn(p0(1)); nout = length(trainY);
    trainX = log(repmat(p0(1),1,nout))';
    iter = 0;
    while iter < niter
        pp = [pp;p];
        trainY = [trainY;fcn(p)];
        trainX = [trainX;repmat(p,1,nout)']; % log
        fn = min(arrayfun(@(i) mean(trainY(i:i+nout-1)),1:nout:length(trainY)-nout+1));
%         [mu,Sigma] = gaussianProcess(trainX,trainY,log(pp));
        mu = arrayfun(@(i) mean(trainY(i:i+nout-1)),1:nout:length(trainY)-nout+1);
        Sigma = arrayfun(@(i) var(trainY(i:i+nout-1)),1:nout:length(trainY)-nout+1);
        Sigma = diag(Sigma+1e-8);
        expectedImprovement = @(p) (fn-interp1(pp,mu,p)).*normcdf((fn-interp1(pp,mu,p))./interp1(pp,diag(Sigma),p))+...
            +interp1(pp,diag(Sigma),p).*normpdf((fn-interp1(pp,mu,p))./interp1(pp,diag(Sigma),p));
        p = fminbnd(expectedImprovement,bnd(1),bnd(2)); % log
        iter = iter+1;
        if ~all(pp-p)
            break
        end
    end
    if all(pp-p)
        pp = [pp;p];
        trainY = [trainY;fcn(p)];
    end
    [error,i] = min(arrayfun(@(i) mean(trainY(i:i+nout-1)),1:nout:length(trainY)-nout+1));
    p = pp(i);
    
    %% plot
%     if length(trainY) ~= length(trainX), pp = pp(1:end-1); end
%     [pp,I] = sort(pp);
%     [mu,Sigma] = gaussianProcess(trainX,trainY,log(pp));
%     pc = [0.025, 0.975]; % percentiles
%     mu = mu(I);
%     Sigma = sqrt(diag(Sigma)); Sigma = Sigma(I);
%     ql = norminv(pc(1),mu,Sigma);
%     qu = norminv(pc(2),mu,Sigma);
%     figure;
%     color = 0.8* [1 1 1];
%     xx = [pp(:);flipud(pp(:))];
%     yy =  [qu(:); flipud(ql(:))];
%     h  = fill(xx,yy,color,'EdgeColor', color); hold on;
%     plot(pp,mu,'LineWidth',2); hold on;
%     set(h,'facealpha',.5);
%     set(gca, 'XScale', 'log')
end

function [m,s] = gpSampler(mu,Sigma,X,xx)
    m = interp1(X,mu,xx);
    s = interp1(X,diag(Sigma),xx);
end

%% Petrov-Galerkin Projection Reduction
function [rSys,fr,Ar,Br,Cr,Er] = PGR(V,W,Ac,Ap,B,C,E,u)
    Acr = W'*Ac*V; Apr = W'*Ap*V;
    Ar = @(p) Acr+p*Apr; Er = W'*E*V;
    Br = W'*B; Cr = C*V;
    rSys = @(p) dss(Ar(p),Br,Cr,0,Er);
    fr = @(t,x,p) Er\(Ar(p)*x + Br*u(t));
end

% Second-order reduction (provides first order output)
function [rSys,fr,Ar,Br,Cr,Er,V] = PGR2(V,W,M,K,a,B2,sig,E,A,B,u)
    r = size(V,2); R = size(V,1)/2;
    V = V(1:size(V,1)/2,1:r); %V = [V,V];
    W = W(1:size(W,1)/2,1:r); %W = [W,W];
    
    Mr = W'*M*V; Kr = W'*K*V;
    Z = zeros(r); I = eye(r);
    A0 = [Z I ; -Kr Z];
    A1 = [Z Z ; Z -Mr];
    A2 = [Z Z ; Z -Kr];
    Ac = A0 + a*A1; Ap = A2;
    Br = [zeros(r,1); W'*B2];
    Cr = [eye(1,R)*V,zeros(1,r)];
    Er = [I Z ; Z Mr];
    
    Ar = @(p) Ac+p*Ap;
    rSys = @(p) dss(Ar(p),Br,Cr,0,Er);
    fr = @(t,x,p) Er\(Ar(p)*x + Br*u(t));
    
    V2 = [];
    for i = 1:size(sig,1)
        V2 = [V2,(sig(i,1)*E-A(sig(i,2)))^2\B];
    end
    V = [V,V2(1:size(V2,1)/2,:)];
end

%% IRKA
function [V,W,sig] = IRKA(A,B,C,E,sig,mu)
%     C = B.';
    if length(mu) == 1 % if only 1 sample, assume concatenation outside func
        p = mu;
        TOL = 1e-6;
        sigo = sig+2*TOL;
        r = length(sig);

        V = []; W = [];
        for i = 1:r
            V = [V,(sig(i)*E-A(p))\B];
            W = [W,(C/(sig(i)*E-A(p))).'];
        end
        W = ((W.'*V)\W.').';
%         W = V;

        iter = 0;
        while norm(sigo-sig,'Inf') > TOL
            if iter > 100
                sig = sig(1:end-1);
                V = V(:,1:end-1); W = W(:,1:end-1);
                r = r-1; iter = 0;
            end
            sigo = sig;
            Ar = full(W.'*A(p)*V); Er = full(W.'*E*V);
            sig = -eig(Ar,Er);% sig = sig(1:length(sig)/2);
%             sig = real(sig)-imag(sig);
            V = []; W = [];
            for i = 1:r
                V = [V,(sig(i)*E-A(p))\B];
                W = [W,(C/(sig(i)*E-A(p))).'];
            end
            W = ((W.'*V)\W.').';
            iter = iter+1;
        end
%         V = V(:,[1,size(V,2)]); W = W(:,[1,size(W,2)]); % TEMPORARY
%         V = orth([real(V)+imag(V)]); % ({+},{,})
%         W = orth([real(W)+imag(W)]); % ({+},{,})
        V = orth([real(V),imag(V)]);
        W = orth([real(W),imag(W)]);
        sig = [sig,p*ones(length(sig),1)];
        
    else % otherwise construct everything internally
        V = []; W = []; sig = [];
        for j = 1:length(mu)
            [Vs,Ws,sigs] = IRKA(A,B,C,E,sig,mu(j));
            V = [V,Vs]; W = [W,Ws]; sig = [sig;[sigs,mu(j)*ones(length(sigs),1)]];
        end
    end
end
