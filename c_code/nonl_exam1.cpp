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
    cout << "Biharmonic operatore for nonlinear function\n\n";                      /* inputs */
    n = 3;
    m = 1;
    d = 4;
    p = 0.5*n*(n+1); // propagate p unit vectors

    cout << "n = "<< n << " p = " << p << "\n";

    /*--------------------------------------------------------------------------*/
    double* xp = new double[n];                        /* allocations and inits */
    double* yp = new double[m];



    adouble* x = new adouble[n];
    adouble* y = new adouble[m];
    adouble v1,v2,v3, v4;
    
    xp[0] = 1.0;
    xp[1] = 2.0;
    xp[2] = 0.5;
    

    /*--------------------------------------------------------------------------*/
    trace_on(1);                                       /* tracing the function */
      y[0] = 1;

      for (i=0; i<n; i++) {
        x[i] <<= xp[i];
      } 

      v1 = x[0]*x[0]*x[0]*x[0];
      v2 = x[1]*x[1]*x[1]*x[1];
      v3 = x[2]*x[2]*x[2]*x[2];
      v4 = v1*v2;
      y[0] = v4*v3;
      y[0] >>= yp[0];
    trace_off();

    /*--------------------------------------------------------------------------*/

    cout <<"Propagate p directions \n";

    double ***XPPP;
    double ***YPPP;
    
    XPPP = new double **[n];
    for (i = 0; i < n; i++) {
      XPPP[i] = new double *[p];
      for (j = 0; j < p; j++) {
        XPPP[i][j] = new double[d];
	XPPP[i][j][0] = 0;
	XPPP[i][j][1] = 0;
      }
    }
    // pure second order derivatives
    for (i = 0; i < n; i++) 
      XPPP[i][i][0] = 1;

    // mixed second order derivatives
      XPPP[0][3][0] = 1;
      XPPP[1][3][0] = 1;
      XPPP[0][4][0] = 1;
      XPPP[2][4][0] = 1;
      XPPP[1][5][0] = 1;
      XPPP[2][5][0] = 1;

    
    YPPP = new double **[1];
    YPPP[0] = new double *[p];
    for (j = 0; j < p; j++)
        YPPP[0][j] = new double[d];

    hov_forward(1,1,n,4,p,xp,XPPP,yp,YPPP);
   
    for(i=0;i<p;i++)
      {
	cout << i << " " <<YPPP[0][i][0] << " " <<YPPP[0][i][1] <<  " " <<YPPP[0][i][2] <<  " " <<YPPP[0][i][3] << "\n";    
      }
    

    /*--------------------------------------------------------------------------*/

     cout <<"hand coded, propagate p directions in standard Taylor arithmetic \n";

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
     x0ts = new double *[p];
     x1ts = new double *[p];
     x2ts = new double *[p];
     v1ts = new double *[p];
     v2ts = new double *[p];
     v3ts = new double *[p];
     v4ts = new double *[p];
     yts = new double *[p];
     
     for (i = 0; i < p; i++) {
       x0ts[i] = new double[4];
       x1ts[i] = new double[4];
       x2ts[i] = new double[4];
       v1ts[i] = new double[4];
       v2ts[i] = new double[4];
       v3ts[i] = new double[4];
       v4ts[i] = new double[4];
       yts[i] = new double[4];
       for (j = 0; j < 4; j++)
	 {
	   x0ts[i][j] = 0; x1ts[i][j] = 0; x2ts[i][j] = 0; v1ts[i][j] = 0; v2ts[i][j] = 0; v3ts[i][j] = 0; v4ts[i][j] = 0;
	 }
     }
     
     // init direction fourth order derivatives
     x0ts[0][0] = 1.0;
     x1ts[1][0] = 1.0; 
     x2ts[2][0] = 1.0;
     x0ts[3][0] = 1.0;
     x1ts[3][0] = 1.0;
     x0ts[4][0] = 1.0;
     x2ts[4][0] = 1.0;
     x1ts[5][0] = 1.0;
     x2ts[5][0] = 1.0;

     // v_1 =  x[0]*x[0]*x[0]*x[0];
     v1ps =  xp[0]*xp[0]*xp[0]*xp[0];
     temps = 1.0/xp[0]; 
     for  (i = 0; i < p; i++) {
       //v_1
       v1ts[i][0] = temps*(4*v1ps*x0ts[i][0]);
       v1ts[i][1] = temps*(4*(v1ts[i][0]*x0ts[i][0]+v1ps*2*x0ts[i][1])-x0ts[i][0]*v1ts[i][0])/2.0;
       v1ts[i][2] = temps*(4*(v1ts[i][1]*x0ts[i][0]+v1ts[i][0]*2*x0ts[i][1]+v1ps*3*x0ts[i][2])-x0ts[i][1]*v1ts[i][0]-2*x0ts[i][0]*v1ts[i][1])/3.0;
       v1ts[i][3] = temps*(4*(v1ts[i][2]*x0ts[i][0]+v1ts[i][1]*2*x0ts[i][1]+v1ts[i][0]*3*x0ts[i][2]+v1ps*4*x0ts[i][3])-x0ts[i][2]*v1ts[i][0]-2*x0ts[i][1]*v1ts[i][1]-3*x0ts[i][0]*v1ts[i][2])/4.0;
     }
     
     // v_2 =  x[1]*x[1]*x[1]*x[1];
     v2ps =  xp[1]*xp[1]*xp[1]*xp[1];
     temps = 1.0/xp[1]; 
     for  (i = 0; i < p; i++) {
       //v_2
       v2ts[i][0] = temps*(4*v2ps*x1ts[i][0]);
       v2ts[i][1] = temps*(4*(v2ts[i][0]*x1ts[i][0]+v2ps*2*x1ts[i][1])-x1ts[i][0]*v2ts[i][0])/2.0;
       v2ts[i][2] = temps*(4*(v2ts[i][1]*x1ts[i][0]+v2ts[i][0]*x1ts[i][1]+v2ps*2*x1ts[i][2])-x1ts[i][1]*v2ts[i][0]-2*x1ts[i][0]*v2ts[i][1])/3.0;
       v2ts[i][3] = temps*(4*(v2ts[i][2]*x1ts[i][0]+v2ts[i][1]*2*x1ts[i][1]+v2ts[i][0]*3*x1ts[i][2]+v2ps*4*x1ts[i][3])-x1ts[i][2]*v2ts[i][0]-2*x1ts[i][1]*v2ts[i][1]-3*x1ts[i][0]*v2ts[i][2])/4.0;
     }
	 

     // v_3 =  x[2]*x[2]*x[2]*x[2];
     v3ps =  xp[2]*xp[2]*xp[2]*xp[2];
     temps = 1.0/xp[2]; 
     for  (i = 0; i < p; i++) {
       //v_3
       v3ts[i][0] = temps*(4*v3ps*x2ts[i][0]);
       v3ts[i][1] = temps*(4*(v3ts[i][0]*x2ts[i][0]+v3ps*2*x2ts[i][1])-x2ts[i][0]*v3ts[i][0])/2.0;
       v3ts[i][2] = temps*(4*(v3ts[i][1]*x2ts[i][0]+v3ts[i][0]*x2ts[i][1]+v3ps*2*x2ts[i][2])-x2ts[i][1]*v3ts[i][0]-2*x2ts[i][0]*v3ts[i][1])/3.0;
       v3ts[i][3] = temps*(4*(v3ts[i][2]*x2ts[i][0]+v3ts[i][1]*2*x2ts[i][1]+v3ts[i][0]*3*x2ts[i][2]+v3ps*4*x2ts[i][3])-x2ts[i][2]*v3ts[i][0]-2*x2ts[i][1]*v3ts[i][1]-3*x2ts[i][0]*v3ts[i][2])/4.0;
     }


     v4ps = v1ps*v2ps;
     for  (i = 0; i < p; i++) {
       v4ts[i][0] = v1ps*v2ts[i][0]+v1ts[i][0]*v2ps;
       v4ts[i][1] = v1ps*v2ts[i][1]+v1ts[i][0]*v2ts[i][0]+v1ts[i][1]*v2ps;
       v4ts[i][2] = v1ps*v2ts[i][2] + v1ts[i][0]*v2ts[i][1] + v1ts[i][1]*v2ts[i][0] + v1ts[i][2]*v2ps ;
       v4ts[i][3] = v1ps*v2ts[i][3] + v1ts[i][0]*v2ts[i][2] + v1ts[i][1]*v2ts[i][1] + v1ts[i][2]*v2ts[i][0]  + v1ts[i][3]*v2ps ;
     }
    
     y[0] = v4ps*v3ps;
     for  (i = 0; i < p; i++) {
       yts[i][0] = v4ps*v3ts[i][0]+v4ts[i][0]*v3ps;
       yts[i][1] = v4ps*v3ts[i][1]+v4ts[i][0]*v3ts[i][0]+v4ts[i][1]*v3ps;
       yts[i][2] = v4ps*v3ts[i][2] + v4ts[i][0]*v3ts[i][1] + v4ts[i][1]*v3ts[i][0] + v4ts[i][2]*v3ps ;
       yts[i][3] = v4ps*v3ts[i][3] + v4ts[i][0]*v3ts[i][2] + v4ts[i][1]*v3ts[i][1] + v4ts[i][2]*v3ts[i][0]  + v4ts[i][3]*v3ps ;
     }

     for  (i = 0; i < p; i++) {
       cout << i << " " << yts[i][0] << " " << yts[i][1] << " " << yts[i][2] << " " << yts[i][3]<< "\n";
     }  
     cout << "\n";
	 
     /*--------------------------------------------------------------------------*/
  
    return 1;
    
}


/****************************************************************************/
/*                                                               THAT'S ALL */
