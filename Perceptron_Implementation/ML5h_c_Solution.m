clear
load MNIST_digit_data


j1=1;
j2=1;

for i=1:15000
  if labels_train(i,1)==2
      train_data_2(j1,:)=images_train(i,:);
       labels_traindata_2(j1,1)=1;
        j1=j1+1;
  end  
    if labels_train(i,1)==8
      train_data_8(j2,:)=images_train(i,:);
      labels_traindata_8(j2,1)=-1;
        j2=j2+1;
    end 
end

images_train1(1:500,:)=train_data_2(1:500,:);
images_train1(501:1000,:)=train_data_8(1:500,:);
labels_train1(1:500,:)=labels_traindata_2(1:500,:);
labels_train1(501:1000,:)=labels_traindata_8(1:500,:);



images_test1(1:500,:)=train_data_2(501:1000,:);
images_test1(501:1000,:)=train_data_8(501:1000,:);
labels_test1(1:500,:)=labels_traindata_2(501:1000,:);
labels_test1(501:1000,:)=labels_traindata_8(501:1000,:);

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
    
for i=1:1000  
    
    pred=dot(images_train1(i,:),weights)+b;
    if pred*labels_train1(i,1)<=0
        weights=weights+images_train1(i,:)*labels_train1(i,1); 
        b=b+labels_train1(i,1);
    end
       
    err=0;

for j=1:1000
    pred=dot(images_test1(j,:),weights)+b;
    if sign(pred)*sign(labels_test1(j,1))<=0
        err=err+1;
    end
end

err_arr(i)=(1000-err)/10;
    
       
end
plot(err_arr);




















