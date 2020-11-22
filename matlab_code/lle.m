% LLE ALGORITHM (using K nearest neighbors)
%
% [Y] = lle(X,K,dmax)
%
% X = data as D x N matrix (D = dimensionality, N = #points)
% K = number of neighbors
% dmax = max embedding dimensionality
% Y = embedding as dmax x N matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Y] = lle(X,K,d)

[D,N] = size(X);
fprintf(1,'LLE running on %d points in %d dimensions\n',N,D);


% STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS 
fprintf(1,'-->Finding %d nearest neighbours.\n',K);

X2 = sum(X.^2,1);  % 对矩阵X中每个元素求平方，并按列求和。
distance = repmat(X2,N,1)+repmat(X2',1,N)-2*X'*X;  %X'表示X的转置。计算xi和xj的距离，表示为distance矩阵

[sorted,index] = sort(distance);  % 对距离排序
neighborhood = index(2:(1+K),:);  % 得到最近邻前k个邻居



% STEP2: SOLVE FOR RECONSTRUCTION WEIGHTS 计算权重
fprintf(1,'-->Solving for reconstruction weights.\n');

if(K>D)  % 如果邻居个数大于样本个数，进行正则化
  fprintf(1,'   [note: K>D; regularization will be used]\n'); 
  tol=1e-3; % regularlizer in case constrained fits are ill conditioned
else
  tol=0;  
end

W = zeros(K,N);  % 初始化参数
for ii=1:N  % 遍历样本
   z = X(:,neighborhood(:,ii))-repmat(X(:,ii),1,K); % shift ith pt to origin  计算样本xi和其每个邻居xij的差： (xi-xij) 
   C = z'*z;                                        % local covariance  % 计算方差
   C = C + eye(K,K)*tol*trace(C);                   % regularlization (K>D)  % 如果k<D，则不需要正则化
   W(:,ii) = C\ones(K,1);                           % solve Cw=1  单位矩阵Ik除以C
   W(:,ii) = W(:,ii)/sum(W(:,ii));                  % enforce sum(w)=1
end;


% STEP 3: COMPUTE EMBEDDING FROM EIGENVECTS OF COST MATRIX M=(I-W)'(I-W)
fprintf(1,'-->Computing embedding.\n');

% M=eye(N,N); % use a sparse matrix with storage for 4KN nonzero elements
M = sparse(1:N,1:N,ones(1,N),N,N,4*K*N); 
for ii=1:N
   w = W(:,ii);
   jj = neighborhood(:,ii);
   M(ii,jj) = M(ii,jj) - w';
   M(jj,ii) = M(jj,ii) - w;
   M(jj,jj) = M(jj,jj) + w*w';
end;

% CALCULATION OF EMBEDDING
options.disp = 0; 
options.isreal = 1; 
options.issym = 1; 
[Y,eigenvals] = eigs(M,d+1,0,options);
Y = Y(:,1:d)'*sqrt(N); % bottom evect is [1,1,1,1...] with eval 0


fprintf(1,'Done.\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% other possible regularizers for K>D
%   C = C + tol*diag(diag(C));                       % regularlization
%   C = C + eye(K,K)*tol*trace(C)*K;                 % regularlization
