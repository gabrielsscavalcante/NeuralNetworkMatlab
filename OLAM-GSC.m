close all; clear all; clc

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

%Iniciando error e saída
error = zeros(1, nLinhas);
Y = zeros(nLinhas, 3);
y = zeros(1, 3);

%Calculando W por 
pseudoinversa = inv((X'*X))*X';
W = pseudoinversa*D;

disp('Valores dos Pesos(W) =');
disp(W)

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

%TERMINAR
Ym = testX*W;
[v,i] = max(Ym');


for i = 1:nLinhas
    u = testX(i,:)*W;
    u = u';
    sigmoid = logsig(u);
    bigNumber = 0;
    
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