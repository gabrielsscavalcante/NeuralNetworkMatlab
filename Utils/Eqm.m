function [ result ] = Eqm(w, d, x)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    tam = size(x);
    p = tam(1);
    eqm = 0;
    
    for i = 1:p, u = x(i,:)*w';
        eqm = eqm + ((d(i,1) - u)*(d(i,1) - u));
    end
    
    result = sqrt(eqm/p);
end

