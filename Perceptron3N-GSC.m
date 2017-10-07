close all; clear all; clc

%Dados gerados em sala
% N = 100;
% var = 0.05
% 
% M1 = [0.3  0.3]
% D1 = repmat(M1,N,1) + var*randn(N,2)
% 
% M2 = [0.5  0.8]
% D2 = repmat(M2,N,1) + var*randn(N,2)
% 
% M3 = [0.8  0.5]
% D3 = repmat(M3,N,1) + var*randn(N,2)
% 
% data = [D1 repmat([1 0 0],N,1);
%         D2 repmat([0 1 0],N,1);
%         D3 repmat([0 0 1],N,1)]

disp('Processo de Treinamento');

%Carregando dados através do iris.data
data = load('iris.data');

%Separando dados aleatóriamente para aprendizagem e teste
data = data(randperm(150),:);
trainingData = data(1:120,:);
testingData = data(121:150,:);

%Inicializando matriz desejada e matriz de padrões
D = trainingData(:,5:7);
X = trainingData(:,1:4);

%Tamnho matriz x
sizeX = size(X);

%Número de linhas e colunas
nLinhas = sizeX(1);
nColunas = sizeX(2);

%Criando o bias e encaixando na matriz X
bias = repmat(-1,nLinhas,1);
X = [bias X];

%Criando os pesos aleatoriamente
W = rand(nColunas+1, 3);

%Taxa de aprendizagem
n = 0.1;

%Iniciando error e saída
error = zeros(1, nLinhas);
Y = zeros(nLinhas, 3);
y = zeros(1, 3);

%Iniciando época
t = 0;
i = 1;
error (i,1) = 1;
while error(i,1) == 1
    error (i,1) = 1;
    
    for i = 1:nLinhas,
        u = X(i,:)*W;
        u = u';
        
         for j = 1:3
            if (u(j,:) >= 0)
                y(:,j) = 1;
            else
                y(:,j) = 0;
            end
        end
        
        dif = D(i,:) - y;
        zero = zeros(1,3);
        
        if (isequal(dif,zero))
            error(i,1) = 0;
            t = t + 1;
        else
            e = D(i,:) - y;
            W = W + (n*X(i,:)'*e);
            t = t + 1;
            error(i,1) = 1;
        end
        
        Y(i,:) = y;
    end
end

disp('Valores dos Pesos(W) =');
disp(W)
disp('Quantidade de Épocas(t) =');
disp(t)

disp('Testando os dados');

testD = testingData(:,5:7);
testX = testingData(:,1:4);

sizeTest = size(testX);
nLinhas = sizeTest(1);
result = zeros(nLinhas, 3);

bias = repmat(-1,nLinhas,1);
testX = [bias testX];

errorTest = 0;
correctTest = 0;
yTest = zeros(nLinhas, 3);

for i = 1:nLinhas
    u = testX(i,:)*W;
    u = u';
    sigmoid = logsig(u)
    bigNumber = 0
    
    for j = 1:3
        if (sigmoid(j,:) >= bigNumber)
            bigNumber = sigmoid(j,:);
        end
    end 
    
    for j = 1:3
        if (sigmoid(j,:) == bigNumber)
            result(i,j) = 1;
        else
            result(i,j) = 0;
        end
    end
    
    disp('Teste');
    disp(result(i,:));
    disp(testD(i,:));
    
    if (isequal(result(i,:),testD(i,:)))
        correctTest = correctTest + 1;
    else
        errorTest = errorTest + 1;
    end
end

media = (correctTest/nLinhas) * 100;

disp('Total de acertos = ');
disp(correctTest);
disp('Total de erros = ');
disp(errorTest);
disp('Média de acertos no teste =');
disp(media);