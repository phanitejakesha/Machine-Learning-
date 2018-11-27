clear
load MNIST_digit_data;
j1=1;
k1=1;
for i=1:60000
  if labels_train(i,1)==1
      train_1(j1,:)=images_train(i,:);
       labels_train1(j1,1)=1;
        j1=j1+1;
  end  
    if labels_train(i,1)==6
      train_6(k1,:)=images_train(i,:);
      labels_train6(k1,1)=-1;
        k1=k1+1;
    end 
end
j1=1;
k1=1;
for i=1:10000
  if labels_test(i,1)==1
      train_1_test(j1,:)=images_test(i,:);
       labels_train1_test(j1,1)=1;
        j1=j1+1;
  end  
    if labels_test(i,1)==6
      train_6_test(k1,:)=images_test(i,:);
      labels_train6_test(k1,1)=-1;
        k1=k1+1;
    end 
end

fimage_train=cat(1,train_1,train_6);
flabel_train=cat(1,labels_train1,labels_train6);

for i=1:500
    fimage_test(i,:)=train_1_test(i,:);
    flabel_test(i,1)=labels_train1_test(i,1);
    fimage_test(i+500,:)=train_6_test(i,:);
    flabel_test(i+500,1)=labels_train6_test(i,1);
end

rand('seed',1);
ind=randperm(size(fimage_train,1));
fimage_train=fimage_train(ind,:);
flabel_train=flabel_train(ind,:);
ind=randperm(size(fimage_test,1));
fimage_test=fimage_test(ind,:);
flabel_test=flabel_test(ind,:);

acc=perceptron1(fimage_train,flabel_train,fimage_test,flabel_test,1);
disp(acc);

function [accuracy] = perceptron1(img_train,lbl_train,img_test,lbl_test,maxiter)
   w=zeros(1,784);
   b=0;
   a=0;   
   for i=1:maxiter
     for j=1:size(img_train)
        a(j)=dot(img_train(j,:),w)+b;
        if a(j)*lbl_train(j,1)<=0
            w=w+img_train(j,:)*lbl_train(j,1); 
            b=b+lbl_train(j,1);
           
        end
      end
   end
   error=0;
   for i=1:1000
      act=dot(img_test(i,:),w)+b;
      if act*lbl_test(i,1)<=0
          error=error+1;
      end
   end
   accuracy=((1000-error)/ts)*100;        
end