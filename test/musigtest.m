function [mu,sig] = musigtest(a,w)
%copied from other func
mu = sum2s(a,w)./w;
sig = zeros(length(a)-w+1,1);

for i = 1:length(mu)
    sig(i) = dot2s(a(i:i+w-1)-mu(i),a(i:i+w-1)-mu(i));
end
sig = 1./sqrt(sig);
end


function [ res ] = sum2s(p,w)
   % based on kahan-babuska and ACCURATE SUM AND DOT PRODUCT by Ogita et al
   
   mlen = length(p)-w+1;
   res = zeros(mlen,1);
   pi_k = p(1);
   sig_k = 0;
   
   for i = 2:w
       [pi_k,q] = TwoSum(pi_k,p(i));
       sig_k = sig_k + q;
   end
   
   res(1) = pi_k + sig_k;
   
   for i = w+1:length(p)
       pi_k_rem = -1*p(i-w);
       [pi_k,q] = TwoSum(pi_k,pi_k_rem);
       sig_k = sig_k + q;
       [pi_k,q] = TwoSum(pi_k,p(i));
       sig_k = sig_k + q;
       res(i-w+1) = pi_k + sig_k;
   end
   
end

%xadd
function [x,y] = TwoSum(a,b)
   x = a+b; 
   z = x-a;
   y = ((a-(x-z))+(b-z));

end
