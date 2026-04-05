%% data preprocessing
clc
clear
% vars -except w1 w2 b1 b2
close all

table= readtable('COVID-19_formatted_dataset.csv');
%rearrange table(:,3) from negative to positive
table = sortrows(table,'SARS_Cov_2ExamResult','ascend');

[r c]=size(table);
neg='negative';
%count # of positive and positive cases
for i=1:r
    if strcmp(char(table2cell(table(i,3))),neg)
        num_p=i;
    end
end
num_n=r- num_p;

%create target value                                              *********
positive=0.1;
negative=-0.1;

T=zeros(1,r);
for i=1:r
    if strcmp(char(table2cell(table(i,3))),neg)
        T(i)=negative;
    else 
        T(i)=positive;
    end
end
%----------------------------------------------------------------
%use 90% of data(465/517;73/81) to train the network
pct=0.7;%--------------------------------------------------------**********

num_pTrain=round(pct*num_p);%=465        (# of negative targets for training)
num_nTrain=round(pct*num_n);%=73         (# of positive targets for training)
num_pTest=num_p-num_pTrain; %517-465=52  (# of negative targets for tesing) 
num_nTest=num_n-num_nTrain; %81-73=8     (# of positive targets for tesing)

%t_train
t_train(1:num_pTrain)=T(1:num_pTrain);                        %negative cases
t_train(num_pTrain+1:num_pTrain+1+num_nTrain-1)=T(num_p+1:num_p+1+num_nTrain-1);%positive cases

% 517-465=52 , 81-73=8

%t_test
t_test(1:num_pTest)=T(num_pTrain+1:517);                    %negative cases
t_test(num_pTest+1:num_pTest+1+num_nTest-1)=T(num_p+1+num_nTrain-1+1:598);  %positive cases
%14-3 
select=[6-3 11-3 14-3 ];%selcet (i-3)col of the table    ***************
num_inputs=length(select);%# of inputs %-----------------------------------

%--------------------------------------------------------------------------
%create input data
P=table2array(table(1:end,4:end))';

%p_train
p_train(1:num_inputs,1:num_pTrain)=P(select,1:num_pTrain); 
p_train(1:num_inputs,num_pTrain+1:num_pTrain+1+num_nTrain-1)=P(select,517+1:517+1+num_nTrain-1); 

%p_test
p_test(1:num_inputs,1:num_pTest)=P(select,num_pTrain+1:num_p); 
p_test(1:num_inputs,num_pTest+1:num_pTest+1+num_nTest-1)=P(select,num_p+1+num_nTrain-1+1:598); 
%------------------------------------------------------------------------
%oversampling
ratio=round(num_pTrain/num_nTrain);

temp=p_train(1:num_inputs,num_pTrain+1:num_pTrain+1+num_nTrain-1);
for i=1:ratio-1
p_train=[p_train temp];
end
clear temp

temp=p_test(1:num_inputs,num_pTest+1:num_pTest+1+num_nTest-1);
for i=1:ratio-1
p_test=[p_test temp];
end
clear temp

temp=t_train(num_pTrain+1:num_pTrain+1+num_nTrain-1);
for i=1:ratio-1
t_train=[t_train temp];
end
clear temp

temp=t_test(num_pTest+1:num_pTest+1+num_nTest-1);
for i=1:ratio-1
t_test=[t_test temp];
end
clear temp
% shuffle training set
idx=randperm(length(p_train));

p_train_shuffled=p_train;
p_train_shuffled(1:num_inputs,idx)=p_train(1:num_inputs,:);

t_train_shuffled(idx)=t_train;

%% 1-10-1
close all
layer1_n=10;
layer2_n=1;

w1 = rand(layer1_n,num_inputs);% size=# of input neurons #of inputs
b1 = rand(layer1_n,1);% len=# of input neurons
w2 = rand(1,layer1_n);% size=# of input neurons #of inputs
b2 = rand(1,1);% len=# of input neurons

% find derivative of f1
syms n
dif_f1=matlabFunction(diff(tansig(n),n));

iter = 1;
MSE(1) = 1;
i = 1;
lr = 0.0001;% learning rate
% th = 0.015;%threshold

while (iter<5000 )
    %step 1, find a1 a2
  for i=1:length(p_train_shuffled)
    n1=w1*p_train_shuffled(:,i) + b1;
    a1 = tansig(n1);
    a2 = w2*a1 + b2;
    err(i) = t_train_shuffled(i) - a2;
    %step 2 find F1' and F2'
    F1=diag(dif_f1(n1));
    s2 = -2*err(i);
    s1 = F1*w2'*s2;
    %step 3 update weights
    w2 = w2 - lr*s2*a1';
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
a_test(j) = w2*tansig(w1*p_test(:,j) + b1) + b2;
end

TN=0;
FN=0;
for j=1:num_pTest
     if a_test(j)<=0
         TN=TN+1;
     else
         FN=FN+1;
     end
end

TP=0;
FP=0;
for j=num_pTest+1:length(t_test)
     if a_test(j)>0
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
a_train(j) = w2*tansig(w1*p_train(:,j) + b1) + b2;
end

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