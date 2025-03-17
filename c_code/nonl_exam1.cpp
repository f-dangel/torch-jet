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
    cout << "Weighted Laplacian for nonlinear function\n\n";                      /* inputs */
    n = 3;
    m = 1;
    d = 2;
    p = 6; // propagate p unit vectors

    /*--------------------------------------------------------------------------*/
    double* xp = new double[n];                        /* allocations and inits */
    double* yp = new double[m];
    double** S = new double*[n];
    double** tensoren;



    adouble* x = new adouble[n];
    adouble* y = new adouble[m];

    for (i=0; i<n; i++) {
        xp[i] = 1.0;
  }
    

    /*--------------------------------------------------------------------------*/
    trace_on(1);                                       /* tracing the function */
      y[0] = 1;

      for (i=0; i<n; i++) {
        x[i] <<= xp[i];
	y[0] = y[0]*x[i]*x[i];
      }
      y[0] = 0.5*y[0];
      y[0] >>= yp[0];
    trace_off();

    /*--------------------------------------------------------------------------*/

    double **H;
    H = myalloc2(n, n);

    hessian(1, n, xp, H);

    printmat(" H", n, n, H);
    printf("\n");
    
    /*--------------------------------------------------------------------------*/

    dim = binomi(n-1+2,2);

    cout << "dim " << dim << "\n";
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
      XPPP[i][i][0] = 2;

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

    hov_forward(1,1,n,2,p,xp,XPPP,yp,YPPP);
   
    for(i=0;i<p;i++)
      {
	cout << i << " " <<YPPP[0][i][0] << " " <<YPPP[0][i][1] << "\n";    
      }
    

    /*--------------------------------------------------------------------------*/
  
    return 1;
    
}


/****************************************************************************/
/*                                                               THAT'S ALL */
