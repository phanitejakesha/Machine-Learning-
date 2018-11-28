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


images_train1(1:500,:)=train_data_1(1:500,:);
images_train1(501:1000,:)=train_data_6(1:500,:);
labels_train1(1:500,:)=labels_traindata_1(1:500,:);
labels_train1(501:1000,:)=labels_traindata_6(1:500,:);

images_test1(1:500,:)=train_data_1(501:1000,:);
images_test1(501:1000,:)=train_data_6(501:1000,:);
labels_test1(1:500,:)=labels_traindata_1(501:1000,:);
labels_test1(501:1000,:)=labels_traindata_6(501:1000,:);

%Randomizing the data 

rand('seed', 1);
inds = randperm(size(images_train1, 1));
images_train1 = images_train1(inds, :);
labels_train1 = labels_train1(inds, :);
inds = randperm(size(images_test1, 1));
images_test1 = images_test1(inds, :);
labels_test1 = labels_test1(inds, :);


  
    acc=zeros(1,1000);
   
    
 [acc1]=SVM_train(images_train1,labels_train1,images_test1,labels_test1,0.01);
 [acc2]=SVM_train(images_train1,labels_train1,images_test1,labels_test1,0.1);   
 [acc3]=SVM_train(images_train1,labels_train1,images_test1,labels_test1,100);   
 [acc4]=SVM_train(images_train1,labels_train1,images_test1,labels_test1,1000);   
 
 
 subplot(2,2,1);
 plot(acc1);
 subplot(2,2,2);
 plot(acc2);
 subplot(2,2,3);
 plot(acc3);
 subplot(2,2,4);
 plot(acc4);
 
    
    
function [acc]=SVM_train(images_train1, labels_train1,images_test1,labels_test1,c)


weights=zeros(1,784);
bias_w=0;
n=100;

    grad=zeros(1,784);
    bias_g=0;  

for i=1:1000
         grad=zeros(1,784);
         bias_g=0; 
         pred=dot(images_train1(i,:),weights)+bias_w;         
        if pred*labels_train1(i,1)<=1
            grad=grad+images_train1(i,:)*labels_train1(i,1); 
            bias_g=bias_g+labels_train1(i,1);
        end    
    r=1/i;    
    grad=grad-c*weights;
    weights=weights+r*grad;
    bias_w=bias_w+r*bias_g; 
    
    succ=0;
    err=0;

    for j=1:1000
        pred=dot(images_test1(j,:),weights)+bias_w;
        if pred*labels_test1(j,1)>=1
            succ=succ+1;   
        else
            err=err+1;       
        end   
       
    end
     acc(1,i)=(succ/1000)*100;
end
end
