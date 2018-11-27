clear
load MNIST_digit_data


j1=1;
j2=1;

for i=1:15000
  if labels_train(i,1)==1
      train_data_1(j1,:)=images_train(i,:);
       labels_traindata_1(j1,1)=1;
        j1=j1+1;
  end  
    if labels_train(i,1)==6
      train_data_6(j2,:)=images_train(i,:);
      labels_traindata_6(j2,1)=-1;
        j2=j2+1;
    end 
end


images_train1(1:5,:)=train_data_1(1:5,:);
images_train1(6:10,:)=train_data_6(1:5,:);
labels_train1(1:5,:)=labels_traindata_1(1:5,:);
labels_train1(6:10,:)=labels_traindata_6(1:5,:);



images_test1(1:500,:)=train_data_1(6:505,:);
images_test1(501:1000,:)=train_data_6(6:505,:);
labels_test1(1:500,:)=labels_traindata_1(6:505,:);
labels_test1(501:1000,:)=labels_traindata_6(6:505,:);


%Randomizing the data 

rand('seed', 1);
inds = randperm(size(images_train1, 1));
images_train1 = images_train1(inds, :);
labels_train1 = labels_train1(inds, :);
inds = randperm(size(images_test1, 1));
images_test1 = images_test1(inds, :);
labels_test1 = labels_test1(inds, :);


weights=zeros(1,784);
b=0;
err_arr=[];

for ep=1:1 %% from 1st epoch the error is not changed and model is constant
    
for i=1:10    
    pred=dot(images_train1(i,:),weights)+b;
    if pred*labels_train1(i,1)<=0
        weights=weights+images_train1(i,:)*labels_train1(i,1); 
        b=b+labels_train1(i,1);
    end
    
end

err=0;

for i=1:1000
    pred=dot(images_test1(i,:),weights)+b;
    if sign(pred)*sign(labels_test1(i,1))<=0
        err=err+1;
    end
end

err_arr(ep)=(1000-err)/10;

end

plot(err_arr);














