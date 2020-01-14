function checkNNGradientsLinear(lambda)

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

input_layer_size = 1;
hidden_layer_size = 5;
num_labels = 1;
m = 5;

Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
X  = debugInitializeWeights(m, input_layer_size - 1);
y  = 0.1 + mod(1:m, num_labels)';

nn_params = [Theta1(:) ; Theta2(:)];


costFuncL = @(p) nnCostFunctionLinear(p, input_layer_size, hidden_layer_size, ...
    X, y, lambda);

[cost, grad] = costFuncL(nn_params);
numgrad = computeNumericalGradient(costFuncL, nn_params);

disp([numgrad grad]);
fprintf(['The above two columns you get should be very similar.\n' ...
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);
diff = norm(numgrad-grad)/norm(numgrad+grad);

fprintf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);
end