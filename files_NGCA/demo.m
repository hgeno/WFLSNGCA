X = clover_pattern(1000); %toy example: 1000 datapoints 2D in a
                  % 4-leaf clover hape distribution 


subplot(1,3,1);

scatter(X(1,:),X(2,:));  %visualization
title('Initial data');

axis equal;

display('Empirical covariance of initial data:');
cov(X')                  %(covariance is close to identity)

scaling = 10.^(-1: 0.2:1 );

for s = 1:length(scaling)
 X(2+s,:) = randn(1,1000)*scaling(s); % add 11 Gaussian noise dimensions
end;                                % with different variances (see scaling)


size(X)                         % check the size of X (1000
                                % points in 13 dimensions)

[q,r] = qr(randn(13,13)); % to be sure we're not cheating, we
                          % "shuffle the cards":
X = q*X;                  % we multiply X by a random orthogonal
                          % matrix
			  % now the initial 2 non-Gaussian dimensions are
                          % hidden in 13D

subplot(1,3,2);

scatter(X(1,:),X(2,:));  %visualization
title('Dimensions 1-2 of shuffled data');
axis equal;


[ngspace,projdata,signalspace] = NGCA(X,[]);  % apply NGCA with
                                              % defaults parameters
					      % (in particular,
                                              % searches for 2
                                              % non-Gaussian dimensions)

ngspace
signalspace                                              
                                              
                                              
subplot(1,3,3);
						 
scatter(projdata(1,:),projdata(2,:));     % we got back the initial data
                                          % ...almost (and up to
                                          % rotation)

title('Retrieved data');
axis equal;
