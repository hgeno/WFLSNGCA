function x = laprnd(m,n,lambda)
%LAPRND is Laplace random variable

if lambda <= 0
  error('lambda must be positive.');
end

y = 2 * rand(m,n) - 1;
x = - sign(y) .* log( 1 - abs(y) ) / lambda;



