clear
load MNIST_digit_data

train=zeros(1000,784);  
labels=zeros(1000,1);
weights=zeros(1,784);
b=0;
a=0;
    
j=1;

for i=1:4740
  if labels_train(i,1)==1
      train(j,:)=images_train(j,:);
       labels(j,1)=labels_train(i,1);
        j=j+1;
  end
  
    if labels_train(i,1)==6
      train(j,:)=images_train(j,:);
       labels(j,1)=-1;
       j=j+1;
    end
 
end

for i=1:1000
    for j=1:784
        a=weights(1,j)*train(1,j) + b;
    end
    
    if labels(j,1)*a<=0
         for j=1:784
          weights(1,j)=weights(1,j)+labels(j,1)*train(1,j);
          b=b+labels(j,1);
          
          
         end
    
    end





end