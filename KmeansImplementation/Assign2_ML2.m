clear
load MNIST_digit_data


j=1;
for i=1:10000
  if labels_train(i,1)==1
      train_data(j,:)=images_train(i,:);
       labels_traindata(j,1)=1;
        j=j+1;
  end  
    if labels_train(i,1)==6
      train_data(j,:)=images_train(i,:);
      labels_traindata(j,1)=-1;
        j=j+1;
    end 
end

images_train=train_data(1:1000,:);
labels_train=labels_traindata(1:1000,:);
images_test=train_data(1000:2000,:);
labels_test=labels_traindata(1000:2000,:);

weights=zeros(1,784);
b=0;




rand('seed', 1); %%just to make all random sequences on all computers the same.
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);
inds = randperm(size(images_test, 1));
images_test = images_test(inds, :);
labels_test = labels_test(inds, :);





for ep=1:10
    
for i=1: 1000   
    
a=weights*transpose(images_train(i,:))+b;

if labels_train(i,1)*a<=0
    
        weights=weights+images_train(i,1)*labels_train(i,:);
        b=b+labels_train(i,1);
    
end
end

end

error=0;

for i=1:1000
    a=dot(weights,images_test(i,:))+b;
    if labels_test(i,1)*a>0
        error=error+1;
    end
end



