function showCorrelationGraphs()

[Id,X,Y] = read_csv('DataTask1/train.csv');

y = Y;

%y = log(y)

figure;

for i = 1:size(X,2)
	x = X(:,i);
	
	%x = sqrt(exp(x));
	
	
	subplot(3,5,i);
	scatter(x, y, [], '.');
	title(sprintf('y vs. x%d',i));
	xlabel(sprintf('x%d',i));
	ylabel('y');
end
