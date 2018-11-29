load MNIST_digit_data;

dimensions = round(logspace(1,log10(500),50));

%To use only 1000 data points.
images_train = images_train(1:1000, :);
labels_train = labels_train(1:1000, :);


for i=1:50
    
mean_train=mean(images_train,2);
X1=images_train;
X1=X1-(mean_train * ones(1,784));
[utrain,strain,vtrain]=svds(X1,dimensions(1,i));

reconstructed_train=(X1*vtrain)*vtrain';

err(1,i)=immse(reconstructed_train,images_train);

end


