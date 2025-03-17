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
    cout << "Computation of Laplacian\n\n";                      /* inputs */
    cout << " Number of indenpendents = ?\n ";
    cin >> n;
    m = 1;
    d = 2;
    p = n; // propagate n unit vectors

    /*--------------------------------------------------------------------------*/
    int* multi = new int[d];                         /* allocations and inits */
    double* xp = new double[n];
    double* yp = new double[m];
    double** S = new double*[n];
    double** A = new double*[n];
    double* test = new double[m];
    double** tensoren;
    adouble* x = new adouble[n];
    adouble* y = new adouble[m];
    adouble temp;

    for (i=0; i<d; i++)
        multi[i] = 0;

    for (i=0; i<n; i++) {
        xp[i] = (i+1.0)/(2.0+i);
        S[i] = new double[p];
	A[i] = new double[n];
        for (j=0; j<n; j++)
	  {
	  S[i][j] = (i==j)?1.0:0.0;
	  }
	for (j=0;j<n;j++)
	  {
	    A[i][j] = i+j;;
	  }
    }
    
    printf("target values: \n");
    for (i=0; i<n; i++)
      {
	  printf(" %f ",2*A[i][i]);
      }
    	printf("\n");


    /*--------------------------------------------------------------------------*/
    trace_on(1);                                       /* tracing the function */
      y[0] = 0;

      for (i=0; i<n; i++) {
        x[i] <<= xp[i];
      } 
      for (int i = 0; i < n; i++) {
        temp = 0.0;
        for (int j = 0; j < n; j++) {
	  temp += A[i][j] * x[j];
        }
         y[0] += x[i] * temp;
       }
    
      y[0] >>= yp[0];
    trace_off();

    /*--------------------------------------------------------------------------*/

    cout <<"Propagate p=n directions\n";
    
    dim = binomi(p+d,d);
    tensoren = myalloc2(m,dim);

    cout <<" d = "<<d<<", dim = "<<dim<<"\n";
    
    tensor_eval(1,m,n,d,p,xp,tensoren,S);

    for(i=1;i<=n;i++)
      {
	multi[0] = i;
	multi[1] = i;

	tensor_value(d, m, test, tensoren, multi);

	cout << i << " " << i << " " <<test[0] << "\n";
    
      }

    
    myfree2(tensoren);

    /*--------------------------------------------------------------------------*/
     cout <<"Propagate p=n directions a lot simpler, Taylor coeffcients scaled by 0.5!\n";

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
	cout << i << " " <<YPPP[0][i][1] << "\n";
    
      }
    /*--------------------------------------------------------------------------*/
    cout << "Now: Weighted Laplacian\n\n";                      /* inputs */

    /*--------------------------------------------------------------------------*/

    /*--------------------------------------------------------------------------*/
    trace_on(2);                                       /* tracing the function */
      y[0] = 1;

      for (i=0; i<n; i++) {
        x[i] <<= 1.0;
	y[0] = y[0]*x[i]*x[i];
      }
      y[0] = 0.5*y[0];
      y[0] >>= yp[0];
    trace_off();

    /*--------------------------------------------------------------------------*/

    double **H;
    H = myalloc2(n, n);

    hessian(2, n, xp, H);

    printmat(" H", n, n, H);
    printf("\n");
  
     
    return 1;
    
}


/****************************************************************************/
/*                                                               THAT'S ALL */
