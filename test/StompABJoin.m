function [MatrixProfile, MPindex, MatrixProfile_row, MPindex_row] = StompABJoin(A, B, SubsequenceLength)

% Compute the join of A and B, return nearest neighbor of every subsequence
% of A in B
% Usage:
% [matrixProfile, matrixProfileIndex] = StompABJoin(A, B, subLen)
% Output:
%     matrixProfile: matrix porfile of the AB-join (vector)
%     matrixProfileIndex: matrix porfile index of the AB-join (vector)
% Input:
%     A: input time series (vector)
%     B: input time series (vector)
%     SubsequenceLength: interested subsequence length (scalar)
% This code is an extension based on the idea of the following paper:
% Yan Zhu, Zachary Zimmerman, Nader Shakibay Senobari, Chin-Chia M. Yeh,
% Gareth Funning, Abdullah Mueen, Philip Brisk, and Eamonn Keogh, "Matrix
% Profile II: Exploiting a Novel Algorithm and GPUs to Break the One
% Hundred Million Barrier for Time Series Motifs and Joins," IEEE International Conference on Data Mining (ICDM), 2016.

%% check input
if SubsequenceLength > length(A)/2
    error('Error: Time series is too short relative to desired subsequence length');
end
if SubsequenceLength < 4
    error('Error: Subsequence length must be at least 4');
end
if length(A) == size(A, 2)
   A = A'; 
end

%% initialization
MatrixProfileLength = length(A) - SubsequenceLength + 1;
MatrixProfile = zeros(MatrixProfileLength, 1);
MPindex = zeros(MatrixProfileLength, 1);

MatrixProfileRowLength = length(B) - SubsequenceLength + 1;
MatrixProfile_row = zeros(MatrixProfileRowLength,1);
MPindex_row = zeros(MatrixProfileRowLength,1);

[XA, nA, sumx2A, sumxA, meanxA, sigmax2A, sigmaxA] = ...
    fastfindNNPre(A, SubsequenceLength);

[XB, nB, sumx2B, sumxB, meanxB, sigmax2B, sigmaxB] = ...
    fastfindNNPre(B, SubsequenceLength);

%% compute the matrix profile
distanceProfile=zeros(MatrixProfileLength,1);
updatePos=false(MatrixProfileLength,1);

% i=1
subsequenceA = A(1:1+SubsequenceLength-1);
subsequenceB = B(1:1+SubsequenceLength-1);
[distanceProfile(:,1) lastz]= fastfindNN(XA, subsequenceB, nA, SubsequenceLength, ...
            sumx2A, sumxA, meanxA, sigmax2A, sigmaxA);
distanceProfile(:,1) = abs(distanceProfile);
[MatrixProfile_row(1) MPindex_row(1)] = min (distanceProfile);
MatrixProfile_row(1)
MPindex_row(1)

[distanceProfilecolumn firstz]= fastfindNN(XB, subsequenceA, nB, SubsequenceLength, ...
            sumx2B, sumxB, meanxB, sigmax2B, sigmaxB);

% evaluate initial matrix profile
MatrixProfile(:) = distanceProfile;
MPindex(:) = 1;

dropvalB=B(1);

for i = 2:MatrixProfileRowLength
    % compute the distance profile
    
    subsequence = B(i:i+SubsequenceLength-1);
    
    lastz(2:(nA-SubsequenceLength+1))=lastz(1:(nA-SubsequenceLength))-A(1:(nA-SubsequenceLength))*dropvalB+A((SubsequenceLength+1):nA)*subsequence(SubsequenceLength);
    %lastz(1)=sum(A(1:SubsequenceLength).*subsequence);
    lastz(1)=firstz(i);
    meany=meanxB(i);
    sigmay=sigmaxB(i);
    distanceProfile(:,1) = sqrt(abs(2*(SubsequenceLength-(lastz-SubsequenceLength*meanxA*meany)./(sigmaxA*sigmay))));
    
    %distanceProfile = sqrt(distanceProfile);
    dropvalB=subsequence(1);
    
    % figure out and store the neareest neighbor
    updatePos(:,1) = distanceProfile < MatrixProfile;
    MPindex(updatePos) = i;
    MatrixProfile(updatePos) = distanceProfile(updatePos);
    
    [MatrixProfile_row(i) MPindex_row(i)] = min (distanceProfile);

    if mod(i,1000)==0
        i
    end
end

% m is winSize
function [X, n, sumx2, sumx, meanx, sigmax2, sigmax] = fastfindNNPre(x, m)
n = length(x);
x(n+1:2*n) = 0;
X = fft(x);
cum_sumx = cumsum(x);
cum_sumx2 =  cumsum(x.^2);
sumx2 = cum_sumx2(m:n)-[0;cum_sumx2(1:n-m)];
sumx = cum_sumx(m:n)-[0;cum_sumx(1:n-m)];
meanx = sumx./m;
sigmax2 = (sumx2./m)-(meanx.^2);
sigmax = sqrt(sigmax2);

% m is winSieze
function [dist lastz] = fastfindNN(X, y, n, m, sumx2, sumx, meanx, sigmax2, sigmax)
%x is the data, y is the query
%y = (y-mean(y))./std(y,1);                      %Normalize the query
y = y(end:-1:1);                                %Reverse the query
y(m+1:2*n) = 0;

%The main trick of getting dot products in O(n log n) time
Y = fft(y);
Z = X.*Y;
z = ifft(Z);

%compute y stats -- O(n)
sumy = sum(y);
sumy2 = sum(y.^2);
meany=sumy/m;
sigmay2 = sumy2/m-meany^2;
sigmay = sqrt(sigmay2);

%computing the distances -- O(n) time
%dist = (sumx2 - 2*sumx.*meanx + m*(meanx.^2))./sigmax2 - 2*(z(m:n) - sumy.*meanx)./sigmax + sumy2;
%dist = 1-dist./(2*m);

dist = 2*(m-(z(m:n)-m*meanx*meany)./(sigmax*sigmay));
dist = sqrt(dist);
lastz=real(z(m:n));