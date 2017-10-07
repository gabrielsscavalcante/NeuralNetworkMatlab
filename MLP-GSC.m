clc;
clear all;
close all;

sizeMatrix = 120;

%Carregando data da Iris e separando amostras
data = load('iris.data');
data = data(randperm(size(data,1)),:);
train = data(1:120,:);
test = data(121:150,:);

%Gerando pesos adicionados com o bias
bias = repmat(-1,sizeMatrix,1);
train(:,1:4) = (train(:,1:4) - repmat(min(train(:,1:4)),120,1))./repmat((max(train(:,1:4))-min(train(:,1:4))),120,1);
test(:,1:4) = (test(:,1:4) - repmat(min(test(:,1:4)),30,1))./repmat((max(test(:,1:4))-min(test(:,1:4))),30,1);
train = [bias train];
test = [bias(1:30) test];

%Separando valor desejado
D = train(:,6:8);
X = train(:,1:5);

%Inicializando valores de erros e épocas
periodError = 0;
period = 0;
numberPeriods = 25000;
Y = zeros(sizeMatrix,3);

%Inicializando neurônios escondidos
hiddenNeurons = size(X',2)/30;
standards = size(X,1);
inputSize = size(X,2); 

alpha    = 0.01;
MSE = zeros(1,numberPeriods);
converge = 0;

while (converge == 0)
    
    hiddenW = (rand(inputSize,hiddenNeurons) - 0.5)/10;   
    W  = (rand(3,hiddenNeurons) - 0.5)/10;
    
    for period = 1:numberPeriods
        for j = 1:sizeMatrix

            x = X(j,:);
            d = D(j,:);

            hiddenU = x*hiddenW;
            hiddenY = (tanh(hiddenU))';
            u = hiddenY'*W';
            y = u;
            error = d - y;

            %Ajustando os pesos da camada de saída
            W = W + (error'*alpha*hiddenY');

            %Ajustando os pesos da camada oculta com a derivada da tangente
            %hiperbólica.
            for i = 1:hiddenNeurons
                delta_hiddenW = ...
                alpha*error*W(:,i)*(1-(hiddenY(i).^2))*x;
                hiddenW(:,i)       = hiddenW(:,i) + delta_hiddenW';
            end
        end

        %Gerando a saída da rede
        Y = W*tanh(X*hiddenW)';
        E = D - Y';
  
        MSE(period) = sum(sum((E' * E)^0.5))/sizeMatrix;
        if MSE(period) < 0.001
            converge = 1;
            break 
        end

    end
end

%Iniciando os testes
testD = test(:,6:8);
testX = test(:,1:5);
testError = 0;
testY = zeros(30,3);
for j = 1:30
    x = testX(j,:);
    d = testD(j,:);
    
    hiddenU = x*hiddenW;
    hiddenY = (tanh(hiddenU))';
    u = hiddenY'*W';
    
    for i = 1:3
        if(u(:,i)> 0)
           testY(j,i) = 1;
        else
          testY(j,i) = 0;
        end
    end
    
    %Conferindo a qual classe a iteração pertence
    if(sum(testY(j,:))> 1)
       if u(1) > u(2) && u(1) > u(3)
           testY(j,:) = [1 0 0];
       elseif u(2) > u(3)
           testY(j,:) = [0 1 0];
       else
           testY(j,:) = [0 0 1];
       end
    end
    
    %Checando se a difernça do erro é 0
    error(j,:) = testD(j,:)-testY(j,:);
    e = sum(abs(error(j,:)));
    if(e ~= 0)
        testError = testError + 1;
    end 
end

disp('Quantidade de erros é de:');
disp(testError);
