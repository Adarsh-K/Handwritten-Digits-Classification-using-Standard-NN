function [J grad] = nnCostFunctionLinear(nn_params,...
    input_layer_size,hidden_layer1_size,hidden_layer2_size,X,y,lambda)

%Rolling nn_params
Theta1=reshape(nn_params(1:hidden_layer1_size*(input_layer_size+1),:),...
    hidden_layer1_size,(input_layer_size+1));
Theta2=reshape(nn_params(hidden_layer1_size*(input_layer_size+1)+1:hidden_layer2_size*(hidden_layer1_size+1),:),...
    hidden_layer2_size,(hidden_layer1_size+1));
Theta3=reshape(nn_params(hidden_layer2_size*(hidden_layer1_size+1)+1:end,:),...
    1,(hidden_layer2_size+1));

%Initializing
m=size(X,1);
J=0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad=zeros(size(Theta2));

%Forward Propogation
a1=[ones(size(X,1),1) X];
z2=a1*Theta1';
a2=tanh(z2);
a2=[ones(size(z2,1),1) a2];
z3=a2*Theta2';
a3=tanh(z3);
a3=[ones(size(z3,1),1) a3];
a4=a3*Theta3';

%Cost
J=(1/(2*m))*((a4-y)'*(a4-y));
h=(lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2))+sum(sum(Theta3(:,2:end).^2)));
J=J+h;

%Gradient
delta4=a4-y;
delta3=(delta4*Theta3(:,2:end)).*tanhGradient(z3);
delta2=(delta3*Theta2(:,2:end)).*tanhGradient(z2);
Theta1_grad=(1/m)*delta2'*a1;
Theta2_grad=(1/m)*delta3'*a2;
Theta3_grad=(1/m)*delta4'*a3;
Theta1_grad=Theta1_grad+(lambda/m)*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad=Theta2_grad+(lambda/m)*[zeros(size(Theta2,1),1) Theta2(:,2:end)];
Theta3_grad=Theta3_grad+(lambda/m)*[zeros(size(Theta3,1),1) Theta3(:,2:end)];
grad=[Theta1_grad(:) ;Theta2_grad(:);Theta3_grad(:)];%UnRolling Theta_grads

end
