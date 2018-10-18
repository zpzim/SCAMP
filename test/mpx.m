function [mp] = mpx(a,minlag,w,thresh)

% matrix profile using cross correlation, 

% depends on files sum2s, musigtest, dot2s
n = length(a);
[mu,sig] = musigtest(a,w);


% differentials have 0 as their first entry. This simplifies index
% calculations slightly and allows us to avoid special "first line"
% handling.
df = [0; (1/2)*(a(1+w:n)-a(1:n-w))];
dg = [0; (a(1+w:n) - mu(2:n-w+1)) + (a(1:n-w)-mu(1:n-w))];
diagmax = length(a)-w+1;
mp = zeros(n-w+1,1);

for diag = minlag+1:diagmax
    c = (sum((a(diag:diag+w-1)-mu(diag)).*(a(1:w)-mu(1))));
    for offset = 1:n-w-diag+2
        c = c + df(offset)*dg(offset+diag-1) + df(offset+diag-1)*dg(offset);
        c_cmp = c*(sig(offset)*sig(offset+diag-1));
        if diag == minlag+1
            cmp(offset, :   ) = [c df(offset) dg(offset+diag-1) df(offset+diag-1) dg(offset)];
        end
        if c_cmp > thresh
            mp(offset) = mp(offset) + 1;
            mp(offset+diag-1) = mp(offset + diag - 1) +  1;
        end
    end
end
% to do ed
end
