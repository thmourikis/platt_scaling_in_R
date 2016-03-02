################################################
# Platt scaling given a prediction
################################################
## Implementation of Platt's scaling to convert decision values of SVM to probabilities
library(dplyr)
plattScaling = function(model, predictions){
  ## Impementation was taken from pseudocode in:
  ## Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods, Platt, 1999
  ## Construct the training set for Platt's sigmoid
  
  decision_values = model$decision.values %>% as.vector()
  labels = model$fitted %>% as.vector()
  df = data.frame(dv=decision_values, lbs = labels)
  
  ## the prediction at the moment is a predict object coming from svm predict
  dv = attr(predictions, "decision.values") %>% as.vector()
  preds = data.frame(dv = dv)
  preds$label = as.vector(predictions)
  rownames(preds) = names(predictions)
  
  ## Input parameters
  out = df$dv
  target = df$lbs
  prior1 = df%>%subset(lbs==TRUE)%>%nrow
  prior0 = df%>%subset(lbs==FALSE)%>%nrow
  ## Output
  ## A, B = parameters of sigmoid
  
  A = 0
  B = log((prior0+1)/(prior1+1))
  hiTarget = (prior1+1)/(prior1+2)
  loTarget = 1/(prior0+2)
  lambda = 1e-3
  olderr = 1e300
  
  
  pp = rep((prior1+1)/(prior0+prior1+2), nrow(df))
  count = 0
  
  for (it in 1:100){
    a=0
    b=0
    c=0
    d=0
    e=0
    
    ## First compute the Hessian & gradient of error function
    ## with respect to A & B
    for (i in 1:length(pp)){
      if (target[i]==TRUE){
        t = hiTarget
      }else{
        t = loTarget
      }
      
      d1 = pp[i]-t
      d2 = pp[i]*(1-pp[i])
      a = a + out[i]*out[i]*d2
      b = b + d2
      c = c + out[i]*d2
      d = d + out[i]*d1
      e = e + d1
    }
    ## If gradient is really tiny, then stop
    if ((abs(d) < 1e-9) & (abs(e) < 1e-9)){break}
    
    oldA = A
    oldB = B
    err = 0
    
    ## Loop until goodness of fit increases
    while(TRUE){
      det = (a+lambda)*(b+lambda)-c*c
      if (det==0){ ## If determinant if Hessian is 0
        ## Increase stabilizer
        lambda = lambda*10
        next
      }
      A = oldA + ((b+lambda)*d-c*e)/det
      B = oldB + ((a+lambda)*e-c*d)/det
      
      ## Now compute the goodness of fit
      err = 0
      for (i in 1:length(pp)){
        p = 1/(1+exp(out[i]*A+B))
        pp[i] = p
        ## At this step, make sure log(0) returns -200
        if (p<=1.383897e-87){
          err = err - t*(-200)+(1-t)*log(1-p)
        }else if (p==1){
          err = err - t*log(p)+(1-t)*(-200)
        }else{
          err = err - t*log(p)+(1-t)*log(1-p)
        }
        if(err==-Inf) browser()
      }
      if (err < olderr*(1+1e-7)){
        lambda = lambda*0.1
        break
      }
      ## Error did not decrease: increase stabilizer by factor of 10
      ## and try again
      lambda = lambda* 10
      if (lambda >= 1e6){ ## something is broken. Give up
        break
      }
    }
    diff = err-olderr
    scale = 0.5*(err+olderr+1)
    if (diff > -1e-3*scale & diff < 1e-7*scale){
      count = count + 1
    }else{
      count = 0
    }
    olderr = err
    if (count==3){
      break
    }
    
    
  }
  
  applyPlatt = function(x){
    p = 1/(1+exp(x*A+B))
  }
  
  preds$prob = apply(preds, 1, function(x) applyPlatt(x[1]))
  return(preds)
}