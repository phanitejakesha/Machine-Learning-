clear
load MNIST_digit_data


j1=1;
j2=1;

for i=1:15000
  if labels_train(i,1)==2
      train_data_1(j1,:)=images_train(i,:);
       labels_traindata_1(j1,1)=1;
        j1=j1+1;
  end  
    if labels_train(i,1)==8
      train_data_6(j2,:)=images_train(i,:);
      labels_traindata_6(j2,1)=-1;
        j2=j2+1;
    end 
end

images_train1(1:500,:)=train_data_1(1:500,:);
images_train1(501:1000,:)=train_data_6(1:500,:);
labels_train1(1:500,:)=labels_traindata_1(1:500,:);
labels_train1(501:1000,:)=labels_traindata_6(1:500,:);



images_test1(1:500,:)=train_data_1(501:1000,:);
images_test1(501:1000,:)=train_data_6(501:1000,:);
labels_test1(1:500,:)=labels_traindata_1(501:1000,:);
labels_test1(501:1000,:)=labels_traindata_6(501:1000,:);


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

for ep=1:40 %% from 26th epoch the error is not changed and model is constant
    
for i=1:1000    
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




weights_p=zeros(1,784);
weights_n=zeros(1,784);



for p=1:784
    if weights(1,p)>=0
        weights_p(1,p)=weights(1,p);
    end
    if weights(1,p)<=0        
        weights_n(1,p)=-weights(1,p);
    end    
end


close all
im = reshape(weights_p(1,:), [28 28]);
in = reshape(weights_n(1,:), [28 28]);
wt=vertcat(im,in);
imshow(wt)




















