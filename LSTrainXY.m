function [weightY]=LSTrainXY(Y,lambda1)
  
      [row,col] = size(Y);
      Il =  ones(col,1);
      In = ones(row,1);
 
      
      rho=10^-3;
      gamma1=zeros(row,1);
      weightY =  (Y'*Y +0.1*eye(col)) \ (Y'*Y);
      weightY(logical(eye(size(weightY)))) = 0;

      max_iter=10;
      convergence1=zeros(max_iter,1);
      convergence2=zeros(max_iter,1);
      epsilon_primal=zeros(max_iter,1);
      epsilon_dual=zeros(max_iter,1);
      Lip_Y = 2*(norm(Y'*Y)^2 + rho*(norm(Y'*Y))^2*norm(Il*Il')^2);
      Lip = sqrt(Lip_Y);
      epsilon_abs=1e-4;
      epsilon_rel=1e-2;
      t=0;
      while(t<max_iter)
       t=t+1;
       [weightY]= proximalGradientY(Y,weightY,gamma1,lambda1,Lip,rho);
       weightY = real(weightY);
       gamma1=gamma1+ rho*((Y*weightY)*Il-In);
       
       %primal residual
       convergence1(t,1) = norm((Y*weightY)*Il-In,'fro');
       %primal epsilon
       epsilon_primal(t,1)=sqrt(row)*epsilon_abs+epsilon_rel*norm((Y*weightY)*Il-In,'fro');
       if(convergence1(t,1)<epsilon_primal)
           fprintf('============ %d XXXYYYADMM´Î %d totalloss=============\n', t, convergence1(t,1));
           break;
       end
      end
       

end