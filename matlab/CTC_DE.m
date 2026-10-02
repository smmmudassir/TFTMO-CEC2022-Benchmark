function [bestX,bestF] = CTC_DE(fun,D,lb,ub,maxFES,seed)
% CTC_DE  R2019a-compatible reference implementation of the development
% Caterpillar Threat-Conditioned Differential Evolution optimizer.
% fun accepts a 1xD row vector and returns scalar fitness.

if isscalar(lb), lb = lb*ones(1,D); end
if isscalar(ub), ub = ub*ones(1,D); end
rng(seed,'twister');

NP0 = max(20,min(180,8*D));
NPmin = 4; H = 6;
X = lb + rand(NP0,D).*(ub-lb);
fit = zeros(NP0,1);
for i=1:NP0, fit(i)=fun(X(i,:)); end
fes = NP0; NP = NP0;
stagn = zeros(NP,1);
MF = 0.5*ones(H,1); MCR = 0.8*ones(H,1); mempos = 1;
A = zeros(0,D);
[bestF,bi]=min(fit); bestX=X(bi,:);
opSucc=ones(3,1); opTrials=3*ones(3,1); gen=0;

while fes < maxFES && NP >= NPmin
    gen=gen+1; progress=fes/maxFES;
    [~,order]=sort(fit);
    ranks=zeros(NP,1);
    ranks(order)=(0:NP-1)'/max(1,NP-1);

    Z=(X-lb)./(ub-lb+eps);
    dist=inf(NP,NP);
    for i=1:NP
        for j=i+1:NP
            d=sqrt(mean((Z(i,:)-Z(j,:)).^2));
            dist(i,j)=d; dist(j,i)=d;
        end
    end
    nn=min(dist,[],2); med=median(nn)+1e-12;
    crowd=max(0,min(1,1-nn/(1.8*med)));
    st=max(0,min(1,stagn/max(6,0.7*sqrt(D)+3)));
    threat=max(0,min(1,0.45*st+0.35*crowd+0.20*ranks));

    freeze=false(NP,1);
    eliteCount=max(1,floor(0.10*NP));
    eidx=order(1:eliteCount);
    freeze(eidx)=(st(eidx)<0.25 & crowd(eidx)<0.55);
    active=find(~freeze);
    saved=NP-numel(active);
    bonus=[];
    if saved>0 && ~isempty(active)
        [~,q]=sort(threat(active),'descend');
        bonus=active(q(1:min(saved,numel(q))));
    end
    targets=[active; bonus(:)];

    candF=inf(NP,1); candX=X; candMeta=zeros(NP,3); has=false(NP,1);
    sF=[]; sCR=[]; improv=[];

    for tt=1:numel(targets)
        if fes>=maxFES, break; end
        i=targets(tt); T=threat(i);
        if T<0.30, op=1; elseif T<0.67, op=2; else, op=3; end
        rates=opSucc./opTrials;
        if rand<0.15 && rates(op)<max(rates), [~,op]=max(rates); end

        k=randi(H); F=positiveCauchy(MF(k)); CR=max(0,min(1,MCR(k)+0.1*randn));
        pmin=2/NP; pmax=0.10+0.10*(1-progress);
        if pmin>=pmax, p=pmax; else, p=pmin+rand*(pmax-pmin); end
        pnum=max(2,min(NP,ceil(p*NP))); pbest=order(randi(pnum));

        ids=setdiff(1:NP,[i pbest]); r1=ids(randi(numel(ids)));
        while true
            rr=randi(NP+size(A,1));
            if rr<=NP
                if rr~=i && rr~=r1, xr2=X(rr,:); break; end
            else
                xr2=A(rr-NP,:); break;
            end
        end
        standard=X(i,:)+F*(X(pbest,:)-X(i,:))+F*(X(r1,:)-xr2);

        if op==1
            mutant=standard;
        elseif op==2
            [~,ix]=sort(dist(pbest,:)); ix=ix(1:max(3,min(7,NP-1)));
            c=mean(X(ix,:),1);
            kap=(0.25+0.75*(1-progress))*(0.5+rand);
            decoy=max(lb,min(ub,X(pbest,:)+kap*(X(pbest,:)-c)));
            mutant=X(i,:)+F*(decoy-X(i,:))+F*(X(r1,:)-xr2);
        else
            [~,ix]=sort(dist(i,:)); ix=ix(1:max(3,min(7,NP-1)));
            c=mean(X(ix,:),1); repel=X(i,:)-c; nr=norm(repel);
            if nr<1e-12, repel=randn(1,D); nr=norm(repel); end
            repel=repel/(nr+eps);
            heavy=max(-5,min(5,trnd_local(2.5)));
            radius=(0.03*(1-progress)+0.002)*mean(ub-lb);
            mutant=standard+heavy*radius*repel;
        end

        lo=mutant<lb; hi=mutant>ub;
        mutant(lo)=(lb(lo)+X(i,lo))/2;
        mutant(hi)=(ub(hi)+X(i,hi))/2;
        mask=rand(1,D)<CR; mask(randi(D))=true;
        trial=X(i,:); trial(mask)=mutant(mask);
        ft=fun(trial); fes=fes+1; opTrials(op)=opTrials(op)+1;

        if ft<candF(i)
            candF(i)=ft; candX(i,:)=trial; candMeta(i,:)=[F CR op]; has(i)=true;
        end
    end

    for i=find(has)'
        if candF(i)<=fit(i)
            oldf=fit(i); oldx=X(i,:);
            X(i,:)=candX(i,:); fit(i)=candF(i); stagn(i)=0;
            if size(A,1)<NP0, A=[A;oldx]; else, A(randi(size(A,1)),:)=oldx; end %#ok<AGROW>
            if fit(i)<oldf
                sF=[sF;candMeta(i,1)]; sCR=[sCR;candMeta(i,2)]; %#ok<AGROW>
                improv=[improv;oldf-fit(i)]; op=candMeta(i,3); %#ok<AGROW>
                opSucc(op)=opSucc(op)+1;
            end
            if fit(i)<bestF, bestF=fit(i); bestX=X(i,:); end
        else
            stagn(i)=stagn(i)+1;
        end
    end

    if ~isempty(improv)
        w=improv/(sum(improv)+eps);
        MF(mempos)=sum(w.*sF.^2)/(sum(w.*sF)+eps);
        if sum(w.*sCR)>0, MCR(mempos)=sum(w.*sCR.^2)/(sum(w.*sCR)+eps); else, MCR(mempos)=0; end
        mempos=mod(mempos,H)+1;
    end

    if mod(gen,20)==0
        opSucc=1+0.8*(opSucc-1); opTrials=3+0.8*(opTrials-3);
    end

    target=round(NPmin+(NP0-NPmin)*(1-fes/maxFES));
    target=max(NPmin,min(NP,target));
    if target<NP
        [~,keep]=sort(fit); keep=keep(1:target);
        X=X(keep,:); fit=fit(keep); stagn=stagn(keep); NP=target;
        if size(A,1)>NP, A=A(randperm(size(A,1),NP),:); end
    end
end
end

function F=positiveCauchy(mu)
for k=1:50
    F=mu+0.1*tan(pi*(rand-0.5));
    if F>0, F=min(1,F); return; end
end
F=max(1e-8,min(1,mu));
end

function x=trnd_local(nu)
% Student-t without Statistics Toolbox: N(0,1)/sqrt(chi2/nu).
z=randn;
chi2=2*randg(nu/2);
x=z/sqrt(chi2/nu);
end
