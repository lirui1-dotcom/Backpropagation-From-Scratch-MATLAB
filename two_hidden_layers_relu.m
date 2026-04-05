%%
close all
layer1_n=4;
layer2_n=2;

positive=0.8;
negative=0;

%4-2 
w1 = rand(layer1_n,num_inputs);% size=# of input neurons X #of inputs
b1 = rand(layer1_n,1);% len=# of input neurons
w2 = rand(layer2_n,layer1_n);% size=# of input neurons #of inputs
b2 = rand(layer2_n,1);% len=# of input neurons
w3 = rand(1,layer2_n);
b3 = rand(1,1);

% find derivative of f1
syms n
dif_f1=matlabFunction(diff(logsig(n),n));

iter = 1;
MSE(1) = 1;
i = 1;
lr = 0.0001;% learning rate
% th = 0.015;%threshold

while (iter<5000 )
    %step 1, find a1 a2
  for i=1:length(p_train_shuffled)
    n1=w1*p_train_shuffled(:,i) + b1;
    a1 = leaky_relu(n1);  
    n2 = w2*a1 + b2;
    a2 = leaky_relu(n2);
    n3 = w3*a2 + b3;
    a3 = logsig(n3);
    err(i) = t_train_shuffled(i) - a3; % a3 is 1x1, output of the network
    %step 2 find F1' and F2'
    F1=diag(df_leaky_relu(n1));
    F2=diag(df_leaky_relu(n2));
    F3=diag(dif_f1(n3));
    s3 = -2*F3*err(i);
%     s3 = -2*err(i);
    s2 = F2*w3'*s3;
    s1 = F1*w2'*s2;
    
    %step 3 update weights
    w3 = w3 -lr*s3*a2';
    b3 = b3 -lr*s3;
    w2 = w2 -lr*s2*a1';
    b2 = b2 -lr*s2;
    w1 = w1 -lr*s1*p_train_shuffled(:,i)';
    b1 = b1 -lr*s1;
  end
    idx=randperm(length(p_train));

    p_train_shuffled=p_train;
    p_train_shuffled(1:num_inputs,idx)=p_train(1:num_inputs,:);

    t_train_shuffled(idx)=t_train;

    iter = iter + 1;
    MSE(iter) = mse(err);
end

for j=1:length(p_test)
% a_test(j) = w3*tansig(w2*tansig(w1*p_test(:,j) + b1) + b2) + b3;
     a_test(j) = logsig(w3*leaky_relu(w2*leaky_relu(w1*p_test(:,j) + b1) + b2) + b3);
end

TN=0;
FN=0;
for j=1:num_pTest
     if a_test(j)<0.5
         TN=TN+1;
     else
         FN=FN+1;
     end
end

TP=0;
FP=0;
for j=num_pTest+1:length(t_test)
     if a_test(j)>=0.5
         TP=TP+1;
     else
         FP=FP+1;
     end
end
Accuracy=(TP+TN)/(TP+TN+FP+FN)
TPR=TP/(TP+FN)
TNR=TN/(TN+FP)
NPV=TN/(TN+FN)
PPV=TP/(TP+FP)
F1_score=2*(PPV*TPR)/(PPV+TPR)
MSE_final=MSE(5000)

figure('Name','Testing set');
j=[1:length(p_test)];
subplot(2,1,1)
stem(j,a_test(j),'LineStyle','none')
hold on
plot(j,t_test(j))
legend('predicted values','actual values')

subplot(2,1,2)
semilogy(MSE)
title("MSE")

for j=1:length(p_train)
% a_train(j) = w3*tansig(w2*tansig(w1*p_train(:,j) + b1) + b2) + b3;
a_train(j) = logsig(w3*leaky_relu(w2*leaky_relu(w1*p_train(:,j) + b1) + b2) + b3);
end


TN=0;
FN=0;
for j=1:num_pTest
     if a_train(j)<0.5
         TN=TN+1;
     else
         FN=FN+1;
     end
end

TP=0;
FP=0;
for j=num_pTest+1:length(t_test)
     if a_train(j)>=0.5
         TP=TP+1;
     else
         FP=FP+1;
     end
end
Accuracy_Train=(TP+TN)/(TP+TN+FP+FN)
TPR_Train=TP/(TP+FN)
TNR_Train=TN/(TN+FP)
NPV_Train=TN/(TN+FN)
PPV_Train=TP/(TP+FP)
F1_score_Train=2*(PPV*TPR)/(PPV+TPR)

figure('Name','Training set');
j=[1:length(p_train)];
subplot(2,1,1)
stem(j,a_train(j),'LineStyle','none')
hold on
plot(j,t_train(j))
legend('predicted values','actual values')

subplot(2,1,2)
semilogy(MSE)
title("MSE")

function x=leaky_relu(x)
    for i=1:length(x)
        if x(i,:)<0
            x(i,:)=0.01*x(i,:);
        end
    end
    
end

function x=df_leaky_relu(x)
    for i=1:length(x)
        if x(i,:)>=0
            x(i,:)=1;
        else
            x(i,:)=0.01;
        end
    end
    
end