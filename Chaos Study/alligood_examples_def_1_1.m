pkg load signal

start_point = 0.23;

g = @(x) 2*x*(1 - x);

number_iterations = 1000;
orbit = zeros(1, number_iterations);

orbit(1) = g(start_point);

for i = 1:(number_iterations - 1)
  orbit(i+1) = g(orbit(i));
endfor

plot(orbit)
