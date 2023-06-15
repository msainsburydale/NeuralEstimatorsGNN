#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <R_ext/Lapack.h>
#include <stdio.h>
#include <R_ext/Print.h>

/*######################################################################################
  ### Bivariate exponent measure and its partial derivatives for Brown-Resnick model ###
  ######################################################################################*/
// Inputs:
// 'Z': data vector
// 'variog': variogram value
// 'output': variable in which the output value of the function will be stored

void VBR(double *Z,double *variog,double *output) {
  double z1=Z[0];
  double z2=Z[1];
  double a=sqrt(*variog);
  double f1= (a/2) - log(z1/z2)/a;
  double f2= (a/2) - log(z2/z1)/a;
  double P1=pnorm(f1,0,1,1,0);
  double P2=pnorm(f2,0,1,1,0);
  *output=P1/z1 + P2/z2;
}

void V1BR(double *Z,double *variog,double *output) {
  double z1=Z[0];
  double z2=Z[1];
  double a=sqrt(*variog);
  double f1= (a/2) - log(z1/z2)/a;
  double f2= (a/2) - log(z2/z1)/a;
  double P1=exp(-0.5*f1*f1)/sqrt(2*M_PI);
  double P2=exp(-0.5*f2*f2)/sqrt(2*M_PI);
  double P3=pnorm(f1,0,1,1,0);
  *output=-(P3+P1/a)/(z1*z1) + P2/(a*z1*z2);
}

void V2BR(double *Z,double *variog,double *output) {
  double z1=Z[0];
  double z2=Z[1];
  double a=sqrt(*variog);
  double f1= (a/2) - log(z1/z2)/a;
  double f2= (a/2) - log(z2/z1)/a;
  double P1=exp(-0.5*f1*f1)/sqrt(2*M_PI);
  double P2=exp(-0.5*f2*f2)/sqrt(2*M_PI);
  double P3=pnorm(f2,0,1,1,0);
  *output=-(P3+P2/a)/(z2*z2) + P1/(a*z1*z2);
}

void V12BR(double *Z,double *variog,double *output) {
  double z1=Z[0];
  double z2=Z[1];
  double a=sqrt(*variog);
  double f1= (a/2) - log(z1/z2)/a;
  double f2= (a/2) - log(z2/z1)/a;
  double P1=exp(-0.5*f1*f1)/sqrt(2*M_PI);
  double P2=exp(-0.5*f2*f2)/sqrt(2*M_PI);
  *output=-(P1*(1-f1/a)/z1 + P2*(1-f2/a)/z2)/(a*z1*z2);
}

void extcoefBR(double *range,double *smooth,double *h,double *output){
    double variog = R_pow((*h)/(*range),(*smooth));
    double Z[2];
    Z[0]=1.0;
    Z[1]=1.0;
    VBR(Z,&variog,output);
}


/*#######################################################
  ### log pairwise likelihood for Brown-Resnick model ###
  #######################################################*/
// Inputs:
// 'range': range parameter
// 'smooth': smoothness parameter
// 'obs': vector of observations
// 'dist': distance matrix
// 'S': number of stations
// 'n': number of time points
// 'hmax': maximum spatial distance used
// 'output': variable in which the output log pairwise likelihood value will be stored

void LogPairwiseLikelihoodBR(double *range,double *smooth, double *obs, double *dist, int *S, int *n, double *hmax, double *output){
    double sum=0.0;
    double v,v1,v2,v12,Z1,Z2,variog,h,contrib;
    double Z[2];
    int i,j,k;
  
    for(i=1;i<(*S);++i){
        for(j=0;j<i;++j){
            h=dist[j*(*S)+i];
            if(h<=(*hmax)){
                variog = 2*R_pow(h/(*range),(*smooth));
                for(k=0;k<(*n);++k){
                    Z1=obs[k*(*S)+i];
                    Z2=obs[k*(*S)+j];
                    
                    Z[0]=Z1;
                    Z[1]=Z2;
                    VBR(Z,&variog,&v);
                    V1BR(Z,&variog,&v1);
                    V2BR(Z,&variog,&v2);
                    V12BR(Z,&variog,&v12);
                    contrib=-v+log(v1*v2-v12);
                    
                    sum+=contrib;
                }
        
            }
    
        }
    }
    
    *output=sum;
}
