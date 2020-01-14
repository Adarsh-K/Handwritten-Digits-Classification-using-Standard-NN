function p = predictLinear(Theta1,...
    Theta2,X)

m=size(X,1);

h1=tanh([ones(m,1) X]*Theta1');
p=[ones(m,1) h1]*Theta2';
