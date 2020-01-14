function [theta] = TrainLinear(initial_nn_params,input_layer_size,...
    hidden_layer1_size,hidden_layer2_size,X,y,lambda)



options = optimset('MaxIter', 200);
costFunctionL = @(p) nnCostFunctionLinear(p, ...
                                   input_layer_size, ...
                                   hidden_layer1_size,hidden_layer2_size, ...
                                   X, y, lambda);

[theta, cost] = fmincg(costFunctionL, initial_nn_params, options);

end