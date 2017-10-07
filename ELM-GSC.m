clear;close all;clc;

% Gerando pontos de 1 a 10 do X do Seno
X = 0.01:0.01:10;
X = X';

% Gerando o Seno de 2x
Y1 = sin(2*X);
Y1 = Y1';

% Adicionando Rúido ao seno
noiseValue = 0.1;
noise = noiseValue*randn(1, length(Y1)) - noiseValue/2;
Y2 = Y1 + noise;
Y2 = Y2';

% Randomizando pontos
dados = [X Y2];
dados = dados(randperm(size(dados,1)),:);

trainSize = 0.8*length(dados);
train = dados(1:trainSize,1);
dTrain = dados(1:trainSize,2);
test = dados(trainSize+1:size(dados,1),1);
dTest = dados(trainSize+1:size(dados,1),2);

%Processo de Treinamento
numberOfHiddenN = 100;
bestQ = 0;
bestQuadraticError = 100;
for q = 1:numberOfHiddenN
   W1 = rand(q,1);
   H = W1 * train';
   H = logsig(H);
   W2 = dTrain' * pinv(H);
        
   U = W1 * test';
   U = logsig(U);
   WFinal = W2 * U;
 
   e = WFinal - dTest';
   quadraticError = sqrt(sum(e.^2));
   if bestQuadraticError > quadraticError
      bestQ = q;
      bestQuadraticError = quadraticError
      bestY = WFinal;
   end
end

disp('Os melhores indices de erro quadrático e com valor:');
disp(bestQ);
disp(bestQuadraticError);

plot(test,bestY, 'g*'); hold on;
plot(test,dTest, 'b*');