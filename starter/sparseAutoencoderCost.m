function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

%forward propagation
% only one hidden Layer,A to save a2(i),in this layer every data example 
%will produce one ai(),so there will be 10000 ai() in every row in A;
m = size(data);
B1 = repmat(b1,1,m(2));
B2 = repmat(b2,1,m(2));
Z = W1*data+B1; 
A2 = sigmoid(Z);%hidden layer active number
Z3 = W2*A2 +B2;
hwb = sigmoid(Z3);%hwb = A3

%forward cost function J(W,b)
dv = hwb - data;
ave_dv = sum(sum(dv.^2)./2)/m(2);%均方差项
weight_decay = lambda/2*(sum(sum(W1.^2))+sum(sum(W2.^2)));%权重衰减项
Jwb = ave_dv +weight_decay;%forward cost founction

%惩罚因子项KL Jsparse
p_real = sum(A2,2)./m(2);
p_exp =  repmat(sparsityParam,hiddenSize,1);

kl = p_exp.*log(p_exp./p_real) +(1-p_exp).*log((1-p_exp)./(1-p_real));
sparse_decay = sum(kl);
cost = Jwb+beta.*sparse_decay;%cost function

%残差
detanl = hwb.*(1-hwb);
residual_nl = -(data-hwb).*detanl;%输出层
detahid = A2.*(1-A2);
pp_real = repmat(p_real,1,m(2));%hiddenSize*m
pp_exp = repmat(p_exp,1,m(2));%hiddenSize*m
residual_hid = ((W2'*residual_nl)+beta.*(-(pp_exp./pp_real)+(1-pp_exp)./(1-pp_real))).*detahid;%隐藏层

% partial derivative
W1grad = residual_hid*data'./m(2)+lambda.*W1;
W2grad = residual_nl*A2'./m(2)+lambda.*W2;
b1grad = sum(residual_hid,2)./m(2);
b2grad = sum(residual_nl,2)./m(2);



























%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

