pkg load signal

r = 0.5; # origem é sorvedouro
##r = 2.8; # origem é fonte

start_point = 0.64;

# Mapa logistico
g = @(x) (r*x.*(1 - x));

bissetriz = @(x) x;

number_iterations = 100;
orbit = zeros(1, number_iterations);

% Setup the plot range
x_vals = linspace(-10, 10, 500);
y_vals = g(x_vals);

figure;

clf;
hold on;

% Plot the function curve and y = x line
plot(x_vals, y_vals, 'b-', 'LineWidth', 2);
plot(x_vals, x_vals, 'r--', 'LineWidth', 1.5);

% Initialize the cobweb trajectory
x_cur = start_point;
plot([x_cur, x_cur], [0, g(x_cur)], 'g-', 'LineWidth', 1); % Initial vertical line

for i = 1:number_iterations
    x_next = g(x_cur);

    % Horizontal line to y = x line
    plot([x_cur, x_next], [x_next, x_next], 'g-', 'LineWidth', 1);

    % Vertical line to the function curve
    plot([x_next, x_next], [x_next, g(x_next)], 'g-', 'LineWidth', 1);

    x_cur = x_next;
end

% Labels and formatting
xlabel('x_n');
ylabel('x_{n+1}');
title(sprintf('Cobweb Plot'));
##legend('f(x)', 'y = x', 'Trajectory', 'Location', 'NorthWest');

xlim([-3 3]);
ylim([-1 1]);
grid on; grid minor;

hold off;
