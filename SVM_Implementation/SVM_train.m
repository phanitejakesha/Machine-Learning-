
function [weights,bias]=SVM_train(images_train, labels_train,x)
rand('seed', 1);
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);

weights=zeros(1,784);
bias_w=0;

for i=1:size(images_train, 1)
         grad=zeros(1,784);
         bias_g=0; 
         pred=dot(images_train(i,:),weights)+bias_w; 
         if labels_train(i,1)==x
                temp=1;
         else
                temp=-1;
         end
         
         if pred*temp<=1
            grad=grad+images_train(i,:)*temp; 
            bias_g=bias_g+temp;
         end    
    r=1/i;    
    grad=grad-0.1*weights;
    weights=weights+r*grad;
    weights=weights/norm(weights);
    bias_w=bias_w+r*bias_g; 
end
bias=bias_w;
end
