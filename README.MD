### Matlab Pseudo Code

#### Multi-Layer Perceptron Neural Network (Backpropagation)
```matlab
% Initialization
1. Initialize weight matrices W1, W2 and biases b1, b2
2. Set up the error threshold threshold_err, learning rate lr
3. iter = 1; MSE(1) = 1; MSE(1) > threshold_err; k = 1

% Iterations
while (MSE(iter)) >= threshold_err
    % Forward propagation
    a1 = logsig(W1 * [x(k+1); x(k)] + b1);
    a2 = W2 * a1 + b2;
    err(k) = x(k+2) - a2;
    
    % Backpropagation
    s2 = -2 * F2' * err(k);
    s1 = F1' * W2' * s2;
    
    % Update weights and biases
    W2 = W2 - lr * s2 * a1';
    b2 = b2 - lr * s2;
    W1 = W1 - lr * s1 * [x(k+1); x(k)]';
    b1 = b1 - lr * s1;
    
    k = k + 1;
    if k > N_seq  % Reset after processing all samples in sequence
        k = 1;
        iter = iter + 1;
        MSE(iter) = mse(err);
    end
end
```

where:
- `mse()` computes the mean square error
- `x(k)` is the input vector at time k
- `a1` is the hidden layer activation (after log-sigmoid)
- `a2` is the output layer prediction
- `err(k)` is the output prediction error
- `s2` and `s1` are the backpropagated error sensitivities for the output and hidden layers, respectively

#### Single-Layer Neuron Network (LMS)
For a single-layer neuron network, backpropagation reduces to the Least Mean Squares (LMS) algorithm. 

```matlab
% Initialization
1. Initialize weight matrix W and bias b
2. Set up the error threshold threshold_err, learning rate lr
3. iter = 1; MSE(1) = 1; MSE(1) > threshold_err;

% Iterations
while MSE(iter) >= threshold_err
    reset error matrix err = 0;
    for k = 1 : N
        err(:, k) = d(k) - W * x(k) - b;
        W = W + 2 * lr * err(:, k) * x(k)';
        b = b + 2 * lr * err(:, k);
    end
    MSE(iter + 1) = mse(err);
    iter = iter + 1;
end
```

where:
- `N` is the number of training samples
- `mse()` computes the mean square error
- `x(k)` is the input vector for sample k
- `d(k)` is the desired (target) output for sample k
- `W` is the weight matrix
- `b` is the bias vector
- `err` is the instantaneous error at each sample

