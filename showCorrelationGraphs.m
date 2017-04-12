function showCorrelationGraphs()

[Id,X,Y] = read_csv('DataTask2/train.csv');


%% this part maps y-values to colors:
y = zeros(size(Y,1),3);
for k = 1:size(Y,1)
	t = Y(k,1)+1;
	y(k, t) = 1;
end


%% this part is for 3d plotting:
figure;
x1 = X(:,2);
x2 = X(:,9);
x3 = X(:,14);
scatter3(x1, x2, x3, [], y, '.');

%% this part is for 2d plots of feature vs. feature:
% 
% for i = 1:size(X,2)
% 	for j = (i+1):size(X,2)
% 		figure;
% 		x1 = X(:,i);
% 		x2 = X(:,j);
% 
% 		
% 
% 		scatter(x1, x2, [], y, '.');
% 		title(sprintf('x%d vs. x%d',i-1,j-1));
% 		xlabel(sprintf('x%d',i-1));
% 		ylabel(sprintf('x%d',j-1));
% 	end
% end
