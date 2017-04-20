% function [X,whiteningMatrix,dewhiteningMatrix] = whitening(Y,rdim)
function [X,whiteningMatrix,Ncond] = whitening(Y,rdim)

% Whitening data X. 
% This code is inspired by the Patcik's code in the ImageICA package.
% (Patrick Hoyer@university of Helsinki.)

if(nargin<2)
    rdim = size(Y,1);
end

% %Calculate the eigenvalues and eigenvectors of covariance matrix.
% % Here, assuming that means of each rows in Y is are zero.
% 

covMatrix = cov(Y');
Ncond = cond(covMatrix);

[E, D]= eig(covMatrix);

%E: the matrix are composed by eigenvectors as the columns
%D: the diagonal components of this matrix are eigenvalue 

[~,order] = sort(diag(D),'descend');

E= E(:,order(1:rdim));

d=diag(D);
d=real(d.^(-0.5));
D=diag(d(order(1:rdim)));

% whiteningMatrix=D*E';
whiteningMatrix=E*D*E'; % shiino
% dewhiteningMatrix=E*D^(-1);

X=whiteningMatrix*Y; 

% whiteningMatrix=covMatrix^(-1/2);
% X=whiteningMatrix*Y; 
