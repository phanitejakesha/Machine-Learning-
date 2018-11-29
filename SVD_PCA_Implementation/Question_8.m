load MNIST_digit_data;

rand('seed', 1); %%just to make all random sequences on all computers the same.
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);


%To use only 1000 data points.
images_train = images_train(1:1000, :);
labels_train = labels_train(1:1000, :);
dimensions = [2,5,10,20,30,50,70,100,150,200,250,300,400,500,748];
   Test_labels = zeros(10,10000);  
   Train_labels = zeros(10, 1000); 
   
    for i=1:1000
            Train_labels(labels_train(i)+1, i) = 1;
    end
 
    for j=1:10000
            Test_labels(labels_test(j)+1, j) = 1;
    end
    

for i=1:15

%Dimension reduction of trainimages
mean_train=mean(images_train,2);
X1=images_train;
X1=X1-(mean_train * ones(1,784));
[utrain,strain,vtrain]=svds(X1,dimensions(1,i));
cimages_train = X1 * vtrain;

%Dimension reduction of testimages
mean_test=mean(images_test,2);
X2=images_test;
X2=X2-(mean_test * ones(1,784));
cimages_test = X2 * vtrain;

   net = patternnet(10);
   net.trainParam.max_fail = 100;
   net.trainParam.epochs=1000;
   net = train(net, cimages_train', Train_labels);
   y = net(cimages_test');
    [c,cm,ind,per] = confusion(Test_labels, y);
        traceVal = trace(cm);
        Acc(1,i) = traceVal/100;        
end
