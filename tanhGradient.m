function t = tanhGradient(z)

t=zeros(size(z));

t1=tanh(z);
t=1-(t1.^2);