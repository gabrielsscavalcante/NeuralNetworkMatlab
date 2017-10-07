close all; clear all; clc

%Valores de entrada
x = [-1 0 0; -1 0 1; -1 1 0; -1 1 1]

%Valores de saída
%d = [-1; -1; -1; 1]
d = [-1; 1; 1; 1];

%Tamnho matriz x
y = size(x);

%Número de linhas e colunas
nLinhas = y(1);
nColunas = y(2);

%Criando os pesos aleatoriamente
w = rand (1, nColunas);

%Taxa de aprendizagem
n = (nLinhas*nColunas)+(nLinhas*nColunas);
n = 1/n;

%Iniciando época
t = 0;
i = 1;

disp('Processo de Treinamento');
for i = 1:nLinhas,
    erro = 1;
    while erro == 1,
        disp(x(i,:));
        disp(w');
        u = x(i,:)*w';
        disp(u);
        if (u >= 0)
            y = 1;
        else
            y = -1;
        end
        
        if (y~=d(i,1))
            w = w + (n*(d(i,1)-y)*x(i,:));
            t = t +1;
            erro = 1;
        else
            erro=0;
            t = t +1;
        end 
    end
    i = i+1;
end

disp('Valores dos Pesos(w) =');
disp(w)
disp('Quantidade de Épocas(t) =');
disp(t)

disp('Testando os dados');

xTest = [-1 0 0; -1 0 1; -1 1 0; -1 1 1]
yTest = size(xTest);

disp(w)

nLinhasTest = yTest(1);
nColunasTest = yTest(2);

for iTest = 1:nLinhasTest
    uTest = xTest(iTest,:)*w';
    if(uTest >= 0)
        valueD = 1
    else
        valueD = -1
    end
end

