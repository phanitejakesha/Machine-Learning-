load MNIST_digit_data;

rand('seed', 1); %%just to make all random sequences on all computers the same.
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);


%To use only 1000 data points.
images_train = images_train(1:4500, :);
labels_train = labels_train(1:4500, :);


mean_train=mean(images_train,2);
X1=images_train;
X1=X1-(mean_train * ones(1,784));
[utrain,strain,vtrain]=svds(X1,50);

mean_test=mean(images_test,2);
X2=images_test;
X2=X2-(mean_test * ones(1,784));
[utest,sest,vtest]=svds(X2,50);


model = svmtrain(labels_train,images_train*vtrain , '-t 0');

[predictedlabel,accuracy,decisionvalues] = svmpredict(labels_test,images_test*vtrain,model);

succ=0;
for i=1:10000
if predictedlabel(i,1)==labels_test(i,1)
    succ=succ+1;
end
end

acc=succ/100;