function [X] = addnoise(S,r,op)
%ADDNOISE
% rで決められた条件数をもつ共分散行列をもつガウスノイズをシグナルに付与する
% input:
%    S  : signal vectors (op.dz*op.n)
%    r  : certain parameter for comtroling the condition number
%    op : options
%         op.dx : data dimension
%         op.dz : dimension of signals
%         op.n  : number of input
% output:
%    X : data added the Gaussian noise(op.dx*op.n)

% generate the Gaussian noise
d_noise = op.dx-op.dz;
mu_noise = zeros(1,d_noise);
if r == 0
    base = ones(1,d_noise);
else
    base = 10.^(-r:(2*r/(d_noise-1)):r);
end
Smat_noise_tmp = diag(base);    
Smat_noise = cat(3,Smat_noise_tmp);
GMMobj_noise = gmdistribution(mu_noise,Smat_noise,1);
N=random(GMMobj_noise,op.n)';

% rotate the noise
I = eye(d_noise);
theta = pi/4;
for i=1:d_noise
    for j=1:d_noise
        if i<j
            R = eye(d_noise);
            R(i,i) = cos(theta); R(i,j) = -sin(theta);
            R(j,i) = sin(theta); R(j,j) = cos(theta);
            I=R*I;
        end
    end
end
N = I*N;

% add noise to signal
X = [S;N];
end

