close all; clear all; clc;

%Inizializando matriz X aleatoriamente a partir da fun��o 3x+5 e seus
%valores desejados. Criado ru�do com intervalo de -1 e 1 para demonstrar no
%gr�fico.
D = [0:1:15];
D = D';
xmin= -2;
xmax = 2;
[nLinhas, nColunas] = size(D);
a = xmin+rand(nLinhas,1) * (xmax-xmin);
randomX = D + a;
X = 3 * randomX + 5
D = 3 * D + 5

%Normaliza��o.
X = ((X-min(X))/(max(X)-min(X)));
D = ((D-min(D))/(max(D)-min(D)));

%Imprimindo pontos X no gr�fico.
plot(X,'.');

%Inicializando os valores aleat�rios dos pesos e criando uma taxa de
%aprendizagem.
w = rand(1, nColunas+1);
n = (nLinhas*nColunas)+(nLinhas*nColunas);
n = 1/n;
E = 0.000005;
t = 0;
dif = 10;
done = false;
Y = zeros(nLinhas,1);
bias = 1;
xBias = zeros(nLinhas,2);

%Aprendizagem Adaline
while done == false
    
    %Adicionando bias na matrix X
    for i = 1:nLinhas
        xBias(i,1) = bias;
        xBias(i,2) = X(i,1);
    end 
    
    %Erro quadr�tico m�dio com os valores antigos ou iniciais.
    eqmOld = Eqm(w, D, xBias);
   
    %Aproximando os valores dos pesos.
    for i = 1:nLinhas, u = xBias(i,:)*w';
       Y(i,1) = u;
       error = (D(i,1) - u);
       w = w + n*error*xBias(i,:);
    end
    
    t = t + 1;
    eqmNew = Eqm(w, D, xBias);
    dif = abs(eqmNew - eqmOld);
    
    clf
    plot(X(:, 1), D(:,1),'r.');
    grid on
    hold on
    reta = xBias*w';
    plot(X,reta,'k');
    
    %Condi��o de parada com o valor do m�dulo da diferen�a dos erros
    %quadr�ticos e um erro preestabelecido.
    if dif <= E
        done = true;
        break;
    end
end

%TESTE

for i = 1:nLinhas, u = X(i,:)*w';
    if u > 0
        y = 1;
    else
        y = -1;
    end
end