load MNIST_digit_data;

dimensions = [2,5,10,20,30,50,70,100,150,200,250,300,400,500,748];
%To use only 1000 data points.
images_train = images_train(1:1000, :);
labels_train = labels_train(1:1000, :);
for i=1:15
mean_train=mean(images_train,2);
X1=images_train;
X1=X1-(mean_train * ones(1,784));
[utrain,strain,vtrain]=svds(X1,dimensions(1,i));

mean_test=mean(images_test,2);
X2=images_test;
X2=X2-(mean_test * ones(1,784));
[utest,sest,vtest]=svds(X2,dimensions(1,i));


model = svmtrain(labels_train,images_train*vtrain , '-t 0');

[predictedlabel,accuracy,decisionvalues] = svmpredict(labels_test,images_test*vtrain,model);
disp(accuracy);
acc(1,i)=accuracy(1,1);

end
