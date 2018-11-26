clear
load MNIST_digit_data

train_data=zeros(1000,784);  
labels_traindata=zeros(1000,1);
test_data=zeros(1000,784);
labels_testdata=zeros(1000,1);

j=1;
for i=1:4740
  if labels_train(i,1)==1
      train_data(j,:)=images_train(i,:);
       labels_traindata(j,1)=1;
        j=j+1;
  end  
    if labels_train(i,1)==6
      train_data(j,:)=images_train(i,:);
      labels_traindata(j,1)=0;
        j=j+1;
    end 
end



j=1;
for i=1:4750
  if labels_test(i,1)==1
      test_data(j,:)=images_test(i,:);
       labels_testdata(j,1)=1;
        j=j+1;
  end  
    if labels_train(i,1)==6
      test_data(j,:)=images_test(i,:);
      labels_testdata(j,1)=0;
        j=j+1;
    end 
end

rand('seed', 1); %%just to make all random sequences on all computers the same.
inds = randperm(size(train_data, 1));
train_data = train_data(inds, :);
labels_traindata = labels_traindata(inds, :);
inds = randperm(size(test_data, 1));
test_data = test_data(inds, :);
labels_testdata = labels_testdata(inds, :);


weights=[zeros(1,784)];
pred=0;

for j=1:1
    
   for i=1:1000 
      
        if dot(train_data(i,:),weights(1:784).')*labels_traindata(i,1)<=0
       
        
            weights=weights+labels_traindata(i,1)*sum(train_data(i,:));

        end               
   end
   
end


error=0;

    for i=1:1000
         if dot(train_data(i,:),weights(1:784).')>0
            pred=1;
        end
        if dot(train_data(i,:),weights(1:784).')<=0
            pred=0;
        end
        
        if ne(labels_traindata(i,1),pred)
        error=error+1;
        end
        
    end
    
display((1000-error)/10);










for i=1:10
  
    activation=dot(train_data(i,:),weights)+b;  

    if labels_traindata(i,:)*activation<=0
        weights=weights+labels_traindata*train_data(i,:);
        b=b+labels_traindata;
    end
    
        
end

















