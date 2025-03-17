#include <cstdlib>
#include <iostream>
#include <math.h>
#include <adolc/adolc.h>


/***************************************************************************/

void printmat(const char *name, int m, int n, double **M) {
  int i, j;

  printf("%s \n", name);
  for (i = 0; i < m; i++) {
    printf("\n %d: ", i);
    for (j = 0; j < n; j++)
      printf(" %10.4f ", M[i][j]);
  }
  printf("\n");
}


/****************************************************************************/
/*                                                                     MAIN */
int main() {
    int i,j,m,n,d,p,dim;

    /*--------------------------------------------------------------------------*/
    cout << "Laplacian for nonlinear function\n\n";                      /* inputs */
    n = 3;
    m = 1;
    d = 2;
    p = n; // propagate n unit vectors

    /*--------------------------------------------------------------------------*/
    int* multi = new int[d];                         /* allocations and inits */
    double* xp = new double[n];
    double* yp = new double[m];
    double** S = new double*[n];
    double* test = new double[m];
    double** tensoren;
    adouble* x = new adouble[n];
    adouble* y = new adouble[m];
    adouble v1,v2,v3,v4;

    for (i=0; i<d; i++)
        multi[i] = 0;

    for (i=0; i<n; i++) {
        xp[i] = (i+1.0)/(2.0+i);
        S[i] = new double[p];
        for (j=0; j<n; j++)
	  {
	  S[i][j] = (i==j)?1.0:0.0;
	  }
    }
    

    /*--------------------------------------------------------------------------*/
    trace_on(1);                                       /* tracing the function */
      y[0] = 0;

      for (i=0; i<n; i++) {
        x[i] <<= xp[i];
      } 

      v1 = cos(x[0]);
      v2 = sin(x[1]);
      v3 = exp(x[2]);
      v4 = v1*v2;
      y[0] = v4+v3;
      y[0] >>= yp[0];
    trace_off();

    /*--------------------------------------------------------------------------*/

    double **H;
    H = myalloc2(n, n);

    hessian(1, n, xp, H);

    printmat(" H", n, n, H);
    printf("\n");
    
    /*--------------------------------------------------------------------------*/

     cout <<"Propagate p=n directions \n";

    double ***XPPP;
    double ***YPPP;
    
    XPPP = new double **[n];
    for (i = 0; i < n; i++) {
      XPPP[i] = new double *[n];
      for (j = 0; j < n; j++) {
        XPPP[i][j] = new double[d];
          XPPP[i][j][0] = (i==j)?1.0:0.0;
          XPPP[i][j][1] = 0;
      }
    }
    YPPP = new double **[1];
    YPPP[0] = new double *[n];
    for (j = 0; j < n; j++)
        YPPP[0][j] = new double[d];

    hov_forward(1,1,n,2,n,xp,XPPP,yp,YPPP);
   
     for(i=0;i<n;i++)
      {
	cout << i << " " <<YPPP[0][i][0] << " " <<2*YPPP[0][i][1] << "\n";
    
      }
     cout << "\n";
    /*--------------------------------------------------------------------------*/

     cout <<"hand coded, propagate just the first direction \n";

     // taylor polynomials of inputs 
     double x0t[2], x1t[2], x2t[2];
     // variables
     double v1p, v2p, v3p, v4p;
     // taylor polynomials attached to them
     double v1t[2], v2t[2], v3t[2], v4t[2];
     // store common results
     double temp;
      // taylor polynomials of output
     double yt[2];
     
    // init direction
     x0t[0] = 1.0; x0t[1] = 0.0; x1t[0] = 0.0; x1t[1] = 0.0; x2t[0] = 0.0; x2t[1] = 0.0;

     v1p = cos(xp[0]);
     temp = sin(xp[0]);
     v1t[0] = -x0t[0]*temp;
     v1t[1] = -x0t[0]*(x0t[0]*v1p)-2*x0t[1]*temp;

     v2p = sin(xp[1]);
     temp = cos(xp[1]);
     v2t[0] = x1t[0]*temp;
     v2t[1] = -x1t[0]*(x1t[0]*v2p)-2*x1t[1]*temp;
     
     v3p = exp(xp[2]);
     v3t[0] = x2t[0]*v3p;
     v3t[1] = x2t[0]*v3t[0]+2*x2t[1]*v3p;

     v4p = v1p*v2p;
     v4t[0] = v1p*v2t[0]+v1t[0]*v2p;
     v4t[1] = v1p*v2t[1]+v1t[0]*v2t[0]+v1t[1]*v2p;

     y[0] = v4p+v3p;
     yt[0] = v3t[0]+v4t[0];
     yt[1] = v3t[1]+v4t[1];
     cout << "y " << yt[0] << " " << yt[1] << "\n";
     cout << "\n";
    /*--------------------------------------------------------------------------*/

     cout <<"hand coded, propagate n directions in standard Taylor arithmetic \n";

     // taylor polynomials of inputs 
     double** x0ts; double** x1ts; double** x2ts;
     // variables
     double v1ps, v2ps, v3ps, v4ps;
     // taylor polynomials attached to them
     double** v1ts; double** v2ts; double** v3ts; double** v4ts;
     // store common results
     double temps, temps1;
      // taylor polynomials of output
     double** yts;

     double Laplaciants;
     
     // allocate memory
     x0ts = new double *[3];
     x1ts = new double *[3];
     x2ts = new double *[3];
     v1ts = new double *[3];
     v2ts = new double *[3];
     v3ts = new double *[3];
     v4ts = new double *[3];
     yts = new double *[3];
     
     for (i = 0; i < 3; i++) {
       x0ts[i] = new double[2];
       x1ts[i] = new double[2];
       x2ts[i] = new double[2];
       v1ts[i] = new double[2];
       v2ts[i] = new double[2];
       v3ts[i] = new double[2];
       v4ts[i] = new double[2];
       yts[i] = new double[2];
     }
     
    
     // init direction
     x0ts[0][0] = 1.0; x0ts[0][1] = 0.0; x0ts[1][0] = 0.0; x0ts[1][1] = 0.0; x0ts[2][0] = 0.0; x0ts[2][1] = 0.0;
     x1ts[0][0] = 0.0; x1ts[0][1] = 0.0; x1ts[1][0] = 1.0; x1ts[1][1] = 0.0; x1ts[2][0] = 0.0; x1ts[2][1] = 0.0;
     x2ts[0][0] = 0.0; x2ts[0][1] = 0.0; x2ts[1][0] = 0.0; x2ts[1][1] = 0.0; x2ts[2][0] = 1.0; x2ts[2][1] = 0.0;
     //

     // v_1 = phi_2(x)
     v1ps = cos(xp[0]);
     temps = sin(xp[0]);
     temps1 = -2*temps;
     for  (i = 0; i < 3; i++) {
       //v_1
       v1ts[i][0] = x0ts[i][0]*(-temps);
       v1ts[i][1] = x0ts[i][0]*(-x0ts[i][0]*v1p)+x0ts[i][1]*temps1;
     }

     v2ps = sin(xp[1]);
     temps = cos(xp[1]);
     temps1 = -2*temps;
     for  (i = 0; i < 3; i++) {
       v2ts[i][0] = x1ts[i][0]*temps;
       v2ts[i][1] = x1ts[i][0]*(-x1ts[i][0]*-v2p)+x1ts[i][1]*temps1;
     }
     
     v3ps = exp(xp[2]);
     temps1 = 2*v3ps;
     for  (i = 0; i < 3; i++) {
       v3ts[i][0] = x2ts[i][0]*v3ps;
       v3ts[i][1] = x2ts[i][0]*v3ts[i][0]+x2ts[i][1]*temps1;
     }
     
     v4ps = v1ps*v2ps;
     for  (i = 0; i < 3; i++) {
       v4ts[i][0] = v1ps*v2ts[i][0]+v1ts[i][0]*v2ps;
       v4ts[i][1] = v1ps*v2ts[i][1]+v1ts[i][0]*v2ts[i][0]+v1ts[i][1]*v2ps;
     }
     
     y[0] = v4ps+v3ps;
     Laplaciants = 0;
     for  (i = 0; i < 3; i++) {
       yts[i][0] = v3ts[i][0]+v4ts[i][0];
       yts[i][1] = v3ts[i][1]+v4ts[i][1];
       Laplaciants += yts[i][1];       
     }  

     for  (i = 0; i < 3; i++) {
       cout << i << " " << yts[i][0] << " " << yts[i][1] << "\n";
     }  
     cout << "\n";

     
     cout << " Laplacian: " << Laplaciants << "\n\n"; 


    /*--------------------------------------------------------------------------*/

     cout <<"hand coded, compute Laplacian  with adapted arithmetic\n";

     // taylor polynomials of inputs (order 1)
     double* x0l1; double* x1l1; double* x2l1;
     double xl2;
     // variables
     double v1l, v2l, v3l, v4l;
     // taylor polynomials attached to them (order 1)
     double* v1l1; double* v2l1; double* v3l1; double* v4l1;
     // Laplacian
     double v1l2; double v2l2; double v3l2; double v4l2;
     // store common results
     double templ;
      // taylor polynomials of output
     double* yl1;
     double yl2;
     
    // allocate memory
    x0l1 = new double[3];
    x1l1 = new double[3];
    x2l1 = new double[3];
    v1l1 = new double[3];
    v2l1 = new double[3];
    v3l1 = new double[3];
    v4l1 = new double[3];
    yl1 = new double[3];
    
    // init direction
     x0l1[0] = 1.0; x0l1[1] = 0.0; x0l1[2] = 0.0;
     x1l1[0] = 0.0; x1l1[1] = 1.0; x1l1[2] = 0.0; 
     x2l1[0] = 0.0; x2l1[1] = 0.0; x2l1[2] = 1.0; 
     xl2 = 0;
     
     v1l = cos(xp[0]);
     templ = sin(xp[0]);
     v1l2 = 0;
     for  (i = 0; i < 3; i++) {
       //first order Taylor
       v1ts[i][0] = x0l1[i]*(-templ);
       // part of second order Taylor
       v1l2 += x0l1[i]*x0l1[i]*(-v1p);
     }
     // remaining part of second order Taylor exploiting linearity
     v1l2 -= 2*templ*xl2;
     
     v2l = sin(xp[1]);
     templ = cos(xp[1]);
     v2l2 = 0;
     for  (i = 0; i < 3; i++) {
       //first order Taylor
       v2l1[i] = x1l1[i]*templ;
       // part of second order Taylor
       v2l2 += x1l1[i]*x1l1[i]*(-v2p);
     }
     // remaining part of second order Taylor exploiting linearity
     v2l2 += (-2*templ)*v1l2;
    
     v3l = exp(xp[2]);
     v3l2 = 0;
     for  (i = 0; i < 3; i++) {
       //first order Taylor
       v3l1[i] = x2l1[i]*v3l;
       // part of second order Taylor
       v3l2 += x2l1[i]*x2l1[i]*v3l1[i];
     }
     // remaining part of second order Taylor exploiting linearity
     v3l2 += 2*v3l*v3l2;
     
     v4l = v1l*v2l;
     v4l2 = 0;
     for  (i = 0; i < 3; i++) {
       //first order Taylor       
       v4l1[i] = v1l*v2l1[i]+v1l1[i]*v2l;
       // part of second order Taylor
       v4l2 += v1l1[i]*v2l1[i];
     }
     // remaining part of second order Taylor exploiting linearity
     v4l2 += v1l*v2l2+v3l2*v2l;
     
     y[0] = v4l+v3l;
     yl2 = 0;     
     for  (i = 0; i < 3; i++) {
       //first order Taylor
       yl1[i] = v3l1[i]+v4l1[i];
       // part of second order Taylor
       // no mixed terms in addition
     }  
     // remaining part of second order Taylor exploiting linearity
       yl2 += v3l2+v4l2;

     cout << " Laplacian: " << Laplaciants << "\n"; 


     /*--------------------------------------------------------------------------*/
  
    return 1;
    
}



/****************************************************************************/
/*                                                               THAT'S ALL */
