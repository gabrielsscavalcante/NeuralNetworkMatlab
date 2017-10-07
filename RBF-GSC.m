clear;close all;clc;

% Gerando valores iniciais
trainSize = 800;
centers = 10;
sigma = 0.05;
maxCenters = 150
maxSigma = 0.5
bestFold = [10000, 0, 0];

%Gerando pontos aleatórios
% Gerando pontos de 1 a 10 do X do Seno
X = 0.01:0.01:10;
X = X';

% Gerando o Seno de 2x
Y1 = sin(2*X);
Y1 = Y1';

% Adicionando Rúido ao seno
nValue = 0.1;
noise = nValue*randn(1, length(Y1)) - nValue/2;
Y2 = Y1 + noise;
Y2 = Y2';

% Randomizando pontos
XY = [X Y2];
XY = XY(randperm(size(XY,1)),:);
X = XY(:,1);
Y2 = XY(:,2);

dados = XY;
train = dados(1:trainSize,1);
trainResult = dados(1:trainSize,2);
test = dados(trainSize+1:size(dados,1),1);
testResult = dados(trainSize+1:size(dados,1),2);
X = train;
Y2 = trainResult;

% Selecionando os centros elatoriamente
for gridCenter = centers : 10 : maxCenters
    inds =randperm(length(X));
    c_inds = inds(1:gridCenter);
    c = X(c_inds, 1);
    for gridSigma = sigma : 0.05 : maxSigma
        
        % Criando folds
        sizeFold = size(XY,1)/5;
        for i = 1 : 5
            fold(i,:,:) = XY( sizeFold*(i-1)+1 : sizeFold*i,:);
        end
        
        mediaFold = 0;
        
        for folds = 1 : 5
            removedFold = folds;
            
            % Usando folds escolhidos para treino
            H = [];
            Yfold = [];
            Xfold = [];
            
            HR = [];
            
            testFold = fold(removedFold,:,1);
            yTestFold = fold(removedFold,:,2);
            
            for j = 1 : size(fold,1)
                if j ~= removedFold
                    for i = 1 : size(fold,2)
                        h = exp(-1/2*(repmat(fold(j,i,1), length(c), 1) - c).^2/gridSigma.^2);
                        H = [H; h']; 
                    end
                    Yfold = [Yfold fold(j,:,2)];
                    Xfold = [Xfold fold(j,:,1)];
                else
                    for i = 1 : size(fold,2)
                        h = exp(-1/2*(repmat(fold(j,i,1), length(c), 1) - c).^2/gridSigma.^2);
                        HR = [HR; h'];
                        
                    end
                end
            end
            
            %Calculando W
            bias = repmat(-1,length(H),1);
            H = [bias H];
            W = (inv((H'*H))*H')*Yfold';
            
            %Testando no fold removido
            bias2 = repmat(-1,length(HR),1);
            HR = [bias2 HR];
            Y3 = W'*HR';
            
            erro = yTestFold-Y3;
            MSE = (erro*erro')^0.5/size(fold(removedFold,:,1),2);
            mediaFold = mediaFold+MSE;
        end
        
        mediaFold = mediaFold/5;
        
        if bestFold(1) > mediaFold
            bestFold = [mediaFold, gridSigma, gridCenter];
        end
        [mediaFold, gridSigma, gridCenter]
        
    end
end
inds =randperm(length(X));
c_inds = inds(1:bestFold(3));
c = X(c_inds, 1);
H = [];
for i = 1 : length(train)
    h = exp(-1/2*(repmat(train(i,:), length(c), 1) - c).^2/bestFold(2).^2);
    H = [H; h'];
end

%Calculando W
bias = repmat(-1,length(H),1);
H = [bias H];
W = (inv((H'*H))*H')*trainResult;

saidaTR = W'*H';
saidaTR = saidaTR';

plot(train,trainResult,'r*');
hold on;
plot(train,saidaTR, 'b*');
hold off;

HT = [];
for i = 1 : length(test)
    h = exp(-1/2*(repmat(test(i,:), length(c), 1) - c).^2/bestFold(2).^2);
    HT = [HT; h'];
end

bias = repmat(-1,length(HT),1);
HT = [bias HT];
Y = W'*HT';

plot(test,testResult,'r*');
hold on;
plot(test,Y', 'b*');
% plot(testFold,yTestFold,'b*');
% hold on
% plot(testFold,Y3, 'r*');
% hold off