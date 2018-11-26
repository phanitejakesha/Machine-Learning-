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
images_test=train_data(1001:2000,:);
labels_test=labels_traindata(1001:2000,:);

weights=zeros(1,784);
b=0;

for ep=1:1
    
for i=1:1000    
    pred=dot(images_train(i,:),weights);
    if pred*labels_train(i,1)<=0
        weights=weights+images_train(i,:)*labels_train(i,1); 
        b=b+labels_train(i,1);
    end
    
end

end

err=0;

for i=1:1000
    pred=dot(images_test(i,:),weights)+b;
    if sign(pred)*sign(labels_test(i,1))<=0
        err=err+1;
    end
end

