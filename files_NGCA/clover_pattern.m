function x = clover_pattern(n);
%
% Draws n 2D data uniformly in a 4 leaf clover shape
%

count = 0;
x = [];

while (count < n)
  d = 2*rand(2,n)-1;
%  index = find( sqrt(sum(d.^2,1)) < sin(4*atan2(d(1,:),d(2,:))));
  index = find( sum(d.^2,1) < sqrt(abs((d(1,:).*d(2,:)))));
  count = count + length(index);
  x = [ x d(:,index)];
end;

x = x(:,1:n)*sqrt(10); %normalization gives covariance close to Id
