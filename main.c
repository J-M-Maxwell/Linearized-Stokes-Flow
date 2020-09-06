#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

/*
int main(argc, char argv[])
 
  Program Solves Stokes equation on a MAC grid using SOR and Red Black ordering

        The stencils for the Red and Black update of U and V at grid point (i, j) relies
        on the neighboring indices (i+1, j), (i-1, j), (i, j+1), and (i, j-1).
        Instead of explicitly using the neighboring grid points in the kernel for each update,
        the kernel takes shifted arrays for U, V, and P as arguements.
        For example a kernel for the U update takes Lu, Ru, Tu, and Bu as arguements where these
        arrays are copies of U shifted one to the left, right, top, and bottom respectively.

inputs: argc should be 6
        argv[1]: N - Size of grid
        argv[2]: mu
        argv[3]: P - Pressure drop
        argv[4]: omega - relaxation coefficient
        argv[5]: tol - tolerance for convergence
        argv[6]: max_iter - max iterations

outputs:
        Final solutions for U, V, and P saved to
        StokesU.out, StokesV.out, and StokesP.out

       Also writes initial conditions, runtime, and number of iterations to Conditions.out
 */


// Declare static variables for the initial conditions and spacial step sizes that are used in the kernels
__device__ static int dev_N;
__device__ static float dev_mu;
__device__ static int dev_P;
__device__ static float dev_omega;
__device__ static double dev_dx;
__device__ static double dev_dy;

// Declare kernels for updating the Red grid pointf of U and V, the black grid points fo U and V, and all grid points of P
__global__ void RedU( double* u, double* Lu, double* Ru, double* Tu, double* Bu, double* p, double* Lp, double* resu );
__global__ void RedV( double* v, double* Lv, double* Rv, double* Tv, double* Bv, double* p, double* Bp,  double* resv );
__global__ void BlackU( double* u, double* Lu, double* Ru, double* Tu, double* Bu, double* p, double* Lp, double* resu );
__global__ void BlackV( double* v, double* Lv, double* Rv, double* Tv, double* Bv, double* p, double* Bp,  double* resv );
__global__ void AllP( double* u, double* Ru, double* v, double* Tv, double* p, double* resp );

// Begin main function
int main(int argc, char* argv[])
{
        // Set intial values
    int N = atoi(argv[1]); // # of grid points
    float mu = atof(argv[2]);
    int P = atoi(argv[3]); // Pressure drop
    float omega = atof(argv[4]); // Relaxation coeeficient
    long double tol = atof(argv[5]); // Tolerance for Convergence
    int maxiter = atoi(argv[6]); // max iterations

    //Start clock
    double time;
    clock_t begin = clock();
    
        // Set spatial step size in both directions
    double dx = 1./(N-1);
    double dy = dx;

        // Allocate memory for arrays of U, V, and P
    double* u = (double*)malloc(N*N*sizeof(double));
    double* v = (double*)malloc(N*N*sizeof(double));
    double* p = (double*)malloc(N*N*sizeof(double));

        // Allocate memory for shifted arrays of U
    double* Lu = (double*)malloc(N*N*sizeof(double));
    double* Ru = (double*)malloc(N*N*sizeof(double));
    double* Tu = (double*)malloc(N*N*sizeof(double));
    double* Bu = (double*)malloc(N*N*sizeof(double));

        // Allocate memory for shifted arrays of V
    double* Lv = (double*)malloc(N*N*sizeof(double));
    double* Rv = (double*)malloc(N*N*sizeof(double));
    double* Tv = (double*)malloc(N*N*sizeof(double));
    double* Bv = (double*)malloc(N*N*sizeof(double));

        // Allocate memory for shifted arrays of P
    double* Lp = (double*)malloc(N*N*sizeof(double));
    double* Bp = (double*)malloc(N*N*sizeof(double));

    double* resu = (double*)malloc(N*N*sizeof(double));
    double* resv = (double*)malloc(N*N*sizeof(double));
    double* resp = (double*)malloc(N*N*sizeof(double));

         // Set initial values of the array to zero
     int i, j, iter;
     for(j = 0; j < N; ++j)
     {
         for(i = 0; i <N; ++i)
         {
             u[j*N + i] = 0.0;
             v[j*N + i] = 0.0;
             p[j*N + i] = 0.0;
         }
     }

     // Dimension for number of blocks
     int GridDim = N/32;

     // Set number of blocks and dimension of threads in each block
     dim3 dimGrid(GridDim, GridDim);
     dim3 dimBlock(32, 32);

    // Declare arrays for U, V, P
     double* dev_u;
     double* dev_v;
     double* dev_p;
    
    // Declare arrays for shifted arrays
     double* dev_Lu;
     double* dev_Ru;
     double* dev_Tu;
     double* dev_Bu;

     double* dev_Lv;
     double* dev_Rv;
     double* dev_Tv;
     double* dev_Bv;

     double* dev_Lp;
     double* dev_Bp;

    // Declare arrays for the residuals
     double* dev_resu;
     double* dev_resv;
     double* dev_resp;
    
    // Allocate memory for the array declared above

    cudaMalloc((void**)&dev_u, N*N*sizeof(double));
    cudaMalloc((void**)&dev_v, N*N*sizeof(double));
    cudaMalloc((void**)&dev_p, N*N*sizeof(double));

    cudaMalloc((void**)&dev_Lu, N*N*sizeof(double));
    cudaMalloc((void**)&dev_Ru, N*N*sizeof(double));
    cudaMalloc((void**)&dev_Tu, N*N*sizeof(double));
    cudaMalloc((void**)&dev_Bu, N*N*sizeof(double));

    cudaMalloc((void**)&dev_Lv, N*N*sizeof(double));
    cudaMalloc((void**)&dev_Rv, N*N*sizeof(double));
    cudaMalloc((void**)&dev_Tv, N*N*sizeof(double));
    cudaMalloc((void**)&dev_Bv, N*N*sizeof(double));

    cudaMalloc((void**)&dev_Lp, N*N*sizeof(double));
    cudaMalloc((void**)&dev_Bp, N*N*sizeof(double));

    cudaMalloc((void**)&dev_resu, N*N*sizeof(double));
    cudaMalloc((void**)&dev_resv, N*N*sizeof(double));
    cudaMalloc((void**)&dev_resp, N*N*sizeof(double));

    cudaMemcpyToSymbol(dev_N, &N, sizeof(int));
    cudaMemcpyToSymbol(dev_mu, &mu, sizeof(float));
    cudaMemcpyToSymbol(dev_P, &P, sizeof(int));
    cudaMemcpyToSymbol(dev_omega, &omega, sizeof(float));
    cudaMemcpyToSymbol(dev_dx, &dx, sizeof(double));
    cudaMemcpyToSymbol(dev_dy, &dy, sizeof(double));
    
    long double residual = 1.0; // maximum residual acrros U, V, and P
    iter = 1; // current iteration

    // Begin while loop
    while( ( iter < maxiter ) && ( residual > tol ) )
    {
        // Set values for the arrays of U, V, and P shifted one to the left or right in the x direction
        // No right shifted array of P is required

        for(j = 0; j < N; ++j)
        {
            for(i = 0; i < N; ++i)
            {
                if(i == 0) // Handle left bound of U, V, and P
                {
                    Lu[j*N + i] = u[j*N + i];
                    Ru[j*N + i] = u[j*N + i+1];

                    Lv[j*N + i] = v[j*N + i];
                    Rv[j*N + i] = v[j*N + i+1];

                    Lp[j*N + i] = 2*P - p[j*N + i];
                }
                else if( i == N-2) // Handle right bound of V and P
                {
                    Lu[j*N + i] = u[j*N + i-1];
                    Ru[j*N + i] = u[j*N + i+1];

                    Rv[j*N + i] = v[j*N + N-2];
                    Lv[j*N + i] = v[j*N + i-1];

                    Lp[j*N + i] = p[j*N + i-1];
                }
                else if ( i == N-1 ) // Handle right bound of U
                {
                    Lu[j*N + i] = u[j*N + i-1];
                    Ru[j*N + i] = u[j*N + N-1];

                    Lv[j*N + i] = 0.0;
                    Rv[j*N + i] = 0.0;

                    Lp[j*N + i] = 0.0;
                }
                else
                {
                    Lu[j*N + i] = u[j*N + i-1];
                    Ru[j*N + i] = u[j*N + i+1];

                    Lv[j*N + i] = v[j*N + i-1];
                    Rv[j*N + i] = v[j*N + i+1];

                    Lp[j*N + i] = p[j*N + i-1];
                }
            }
        }

        // Set values for the arrays of U, V, and P shifted one Up or Down in the y direction
        // No Up shifted array of P is required

        for( j = 0; j < N; ++ j)
        {
            for(i = 0; i <N; ++i)
            {
                if( j == 0) // Handle lower bound of U, V, and P
                {
                    Bu[j*N + i] = -u[j*N + i];
                    Tu[j*N + i] = u[(j+1)*N + i];

                    Bv[j*N + i] = 0.0;
                    Tv[j*N + i] = v[(j+1)*N + i];

                    Bp[j*N + i] = 0.0;
                }
                else if ( j == N-2 ) // Handle upper bound of U and P
                {
                    Bu[j*N + i] = u[(j-1)*N + i];
                    Tu[j*N + i] = -u[(N-2)*N + i];

                    Bv[j*N + i] = v[(j-1)*N + i];
                    Tv[j*N + i] = v[(j+1)*N + i];

                    Bp[j*N + i] = p[(j-1)*N + i];
                }
                else if ( j == N-1 ) //Handle upper bound of V
                {
                    Bu[j*N + i] = 0.0;
                    Tu[j*N + i] = 0.0;

                    Bv[j*N + i] = v[(j-1)*N + i];
                    Tv[j*N + i] = 0.0;

                    Bp[j*N + i] = 0.0;
                }
                else
                {
                    Bu[j*N + i] = u[(j-1)*N + i];
                    Tu[j*N + i] = u[(j+1)*N + i];

                    Bv[j*N + i] = v[(j-1)*N + i];
                    Tv[j*N + i] = v[(j+1)*N + i];

                    Bp[j*N + i] = p[(j-1)*N + i];
                }
            }
        }
        
        // Copy arrays of U, V, and P to device
        cudaMemcpy(dev_u, u, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_v, v, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_p, p, N*N*sizeof(double), cudaMemcpyHostToDevice);

        // Copy shifted arrays of U to device
        cudaMemcpy(dev_Lu, Lu, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Ru, Ru, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Tu, Tu, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Bu, Bu, N*N*sizeof(double), cudaMemcpyHostToDevice);

        // Coppy shifted arrays of V to device
        cudaMemcpy(dev_Lv, Lv, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Rv, Rv, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Tv, Tv, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Bv, Bv, N*N*sizeof(double), cudaMemcpyHostToDevice);

        // Copy shifted arrays of P to device
        cudaMemcpy(dev_Lp, Lp, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Bp, Bp, N*N*sizeof(double), cudaMemcpyHostToDevice);

        // Coppy arrays of U and V residuals to device
        cudaMemcpy(dev_resu, resu, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_resv, resv, N*N*sizeof(double), cudaMemcpyHostToDevice);

        // Execute Red U update
        RedU<<<dimGrid, dimBlock>>>(dev_u, dev_Lu, dev_Ru, dev_Tu, dev_Bu, dev_p, dev_Lp, dev_resu);

        // Execute Red V update
        RedV<<<dimGrid, dimBlock>>>(dev_v, dev_Lv, dev_Rv, dev_Tv, dev_Bv, dev_p, dev_Bp, dev_resv);

        // Synchronize before proceeding
        cudaDeviceSynchronize();

        // Copy U, V, and P back to host
        cudaMemcpy(u, dev_u, N*N*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(v, dev_v, N*N*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(p, dev_p, N*N*sizeof(double), cudaMemcpyDeviceToHost);

        // Set values for the arrays of U, V, and P shifted one to the left or right in the x direction
        // No right shifted array of P is required
        for(j = 0; j < N; ++j)
        {
            for(i = 0; i < N; ++i)
            {
                if(i == 0)
                {
                    Lu[j*N + i] = u[j*N + i];
                    Ru[j*N + i] = u[j*N + i+1];

                    Lv[j*N + i] = v[j*N + i];
                    Rv[j*N + i] = v[j*N + i+1];

                    Lp[j*N + i] = 2*P - p[j*N + i];
                }
                else if ( i == N-2)
                {
                    Lu[j*N + i] = u[j*N + i-1];
                    Ru[j*N + i] = u[j*N + i+1];

                    Rv[j*N + i] = v[j*N + N-2];
                    Lv[j*N + i] = v[j*N + i-1];

                    Lp[j*N + i] = p[j*N + i-1];
                }
                else if ( i == N-1 )
                {
                    Lu[j*N + i] = u[j*N + i-1];
                    Ru[j*N + i] = u[j*N + N-1];

                    Lv[j*N + i] = 0.0;
                    Rv[j*N + i] = 0.0;

                    Lp[j*N + i] = 0.0;
                }
                else
                {
                    Lu[j*N + i] = u[j*N + i-1];
                    Ru[j*N + i] = u[j*N + i+1];

                    Lv[j*N + i] = v[j*N + i-1];
                    Rv[j*N + i] = v[j*N + i+1];

                    Lp[j*N + i] = p[j*N + i-1];
                }
            }
        }

        // Set values for the arrays of U, V, and P shifted one Up or Down in the y direction
        // No Up shifted array of P is required
        for( j = 0; j < N; ++ j)
         {
             for(i = 0; i <N; ++i)
             {
                 if( j == 0) // Handle lower bound of U, V, and P
                 {
                     Bu[j*N + i] = -u[j*N + i];
                     Tu[j*N + i] = u[(j+1)*N + i];
                     
                     Bv[j*N + i] = 0.0;
                     Tv[j*N + i] = v[(j+1)*N + i];
                     
                     Bp[j*N + i] = 0.0;
                 }
                 else if ( j == N-2 ) // Handle upper bound of U and P
                 {
                     Bu[j*N + i] = u[(j-1)*N + i];
                     Tu[j*N + i] = -u[(N-2)*N + i];
                     
                     Bv[j*N + i] = v[(j-1)*N + i];
                     Tv[j*N + i] = v[(j+1)*N + i];
                     
                     Bp[j*N + i] = p[(j-1)*N + i];
                 }
                 else if ( j == N-1 ) // Handle upper bound of V
                 {
                     Bu[j*N + i] = 0.0;
                     Tu[j*N + i] = 0.0;
                     
                     Bv[j*N + i] = v[(j-1)*N + i];
                     Tv[j*N + i] = 0.0;
                     
                     Bp[j*N + i] = 0.0;
                 }
                 else
                 {
                     Bu[j*N + i] = u[(j-1)*N + i];
                     Tu[j*N + i] = u[(j+1)*N + i];
                     
                     Bv[j*N + i] = v[(j-1)*N + i];
                     Tv[j*N + i] = v[(j+1)*N + i];
                     
                     Bp[j*N + i] = p[(j-1)*N + i];
                 }
             }
         }
                
        // Copy arrays of U, V, and P back to device
        cudaMemcpy(dev_u, u, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_v, v, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_p, p, N*N*sizeof(double), cudaMemcpyHostToDevice);

        // Copy shifted arrays of U back to device
        cudaMemcpy(dev_Lu, Lu, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Ru, Ru, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Tu, Tu, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Bu, Bu, N*N*sizeof(double), cudaMemcpyHostToDevice);

        // Coppy shifted arrays of V back to device
        cudaMemcpy(dev_Lv, Lv, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Rv, Rv, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Tv, Tv, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Bv, Bv, N*N*sizeof(double), cudaMemcpyHostToDevice);

        // Coppy shifted arrays of P back to device
        cudaMemcpy(dev_Lp, Lp, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Bp, Bp, N*N*sizeof(double), cudaMemcpyHostToDevice);

        // Execute Black U update
        BlackU<<<dimGrid, dimBlock>>>(dev_u, dev_Lu, dev_Ru, dev_Tu, dev_Bu, dev_p, dev_Lp, dev_resu);

        // Execute Black V update
        BlackV<<<dimGrid, dimBlock>>>(dev_v, dev_Lv, dev_Rv, dev_Tv, dev_Bv, dev_p, dev_Bp, dev_resv);

        // Synchronize before proceeding
        cudaDeviceSynchronize();

        // Copy U, V, and P back to host
        cudaMemcpy(u, dev_u, N*N*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(v, dev_v, N*N*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(p, dev_p, N*N*sizeof(double), cudaMemcpyDeviceToHost);

        // Coppy U and V residuals back to host
        cudaMemcpy(resu, dev_resu, N*N*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(resv, dev_resv, N*N*sizeof(double), cudaMemcpyDeviceToHost);

        // Set values for the arrays of U shifted one to the right in the x direction
        // Only right shifted U is required to update P
        for(j = 0; j < N; ++ j)
        {
            for(i = 0; i < N; ++i)
            {
                if(i == 0)
                {
                    Ru[j*N + i] = u[j*N + i+1];
                }
                else if ( i == N-1 )
                {
                    Ru[j*N + i] = u[j*N + N-1];
                }
                else
                {
                    Ru[j*N + i] = u[j*N + i+1];
                }
            }
        }

        // Set values of the arrays of V shifted one up in the y direction
        // Only up shifted V is require to update P
        for( j = 0; j < N; ++ j)
        {
            for(i = 0; i <N; ++i)
            {
                if( j == 0)
                {
                    Tv[j*N + i] = v[(j+1)*N + i];
                }
                else if ( j == N-2 )
                {
                    Tv[j*N + i] = v[(j+1)*N + i];;
                }
                else if ( j == N-1 )
                {
                    Tv[j*N + i] = 0.0;
                }
                else
                {
                    Tv[j*N + i] = v[(j+1)*N + i];
                }
            }
        }
        
        // Copy arrays of U, V, and P back to device
        cudaMemcpy(dev_u, u, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_v, v, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_p, p, N*N*sizeof(double), cudaMemcpyHostToDevice);

        // Copy shifted U and V to device
        cudaMemcpy(dev_Ru, Ru, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Tv, Tv, N*N*sizeof(double), cudaMemcpyHostToDevice);

        // Copy array of P residuals to device
        cudaMemcpy(dev_resp, resp, N*N*sizeof(double), cudaMemcpyHostToDevice);

        // Execute P udate for all grid points
        AllP<<<dimGrid, dimBlock>>>(dev_u, dev_Ru, dev_v, dev_Tv, dev_p, dev_resp);

        // synchronize before proceeding
        cudaDeviceSynchronize();

        // Copy aarrays of U, V, and P back to host
        cudaMemcpy(u, dev_u, N*N*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(v, dev_v, N*N*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(p, dev_p, N*N*sizeof(double), cudaMemcpyDeviceToHost);

        // Copy shifted U and V back to host
        cudaMemcpy(Ru, dev_Ru, N*N*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(Tv, dev_Tv, N*N*sizeof(double), cudaMemcpyDeviceToHost);

        // Copy array of P residuals back to host
        cudaMemcpy(resp, dev_resp, N*N*sizeof(double), cudaMemcpyDeviceToHost);

        // place holder for max residual of current update
        long double maxres = 0.0;

        // Find the maximum residual
        for( j = 0; j < N; ++j)
        {
            for( i = 0; i < N; ++i)
            {
                if( abs(resu[j*N + i]) > maxres )
                        maxres = abs(resu[j*N + i]);
                if( abs(resv[j*N + i]) > maxres )
                        maxres = abs(resv[j*N + i]);
                if( abs(resp[j*N + i]) > maxres )
                        maxres = abs(resp[j*N + i]);
            }
        }

        // Set residual to current max residual
        residual = maxres;
        
        // Update iteration count
        iter += 1;

    } // End while loop

        // Declare arrays for the final solution,
        // the current u, v, and p are too large to write out
        double ufin[(N-1)*N];
        double vfin[(N-1)*N];
        double pfin[(N-1)*(N-1)];

    // Set ufin to write out
    for(j = 0; j < N-1; ++j)
    {
        for(i = 0; i < N; ++ i)
        {
            ufin[j*(N) + i] = u[j*N + i];
        }
    }

    // Set vfin to write out
    for(j = 0; j < N; ++j)
    {
        for(i = 0; i < N-1; ++ i)
        {
            vfin[j*(N-1) + i] = v[j*N + i];
        }
    }

    // Set pfin to write out
    for(j = 0; j < N-1; ++j)
    {
        for(i = 0; i < N-1; ++ i)
        {
            pfin[j*(N-1) + i] = p[j*N + i];
        }
    }

    // End clock and find runtime
    time_t end = clock(); // End timer
    time = ((double)(end - begin))/CLOCKS_PER_SEC; // Recored run time

    FILE *fp;
    fp = fopen("StokesU.out", "w"); // open output file as append
    fwrite(ufin, sizeof(double), N*(N-1), fp); // Append to output file
    fclose(fp); // Close output file

    fp = fopen("StokesV.out", "w"); // open output file as append
    fwrite(vfin, sizeof(double), N*(N-1), fp); // Append to output file
    fclose(fp); // Close output file

    fp = fopen("StokesP.out", "w"); // open output file as append
    fwrite(pfin, sizeof(double), (N-1)*(N-1), fp); // Append to output file
    fclose(fp); // Close output file

    fp = fopen("Time.out", "w"); // open output file as append
    fwrite(&time, sizeof(double), 1, fp); // Append to output file
    fclose(fp); // Close output file

    // cudaFree all U related arrays
    cudaFree(dev_u);
    cudaFree(dev_resu);
    cudaFree(dev_Lu);
    cudaFree(dev_Ru);
    cudaFree(dev_Tu);
    cudaFree(dev_Bu);

    // cudaFree all V related arrays
    cudaFree(dev_v);
    cudaFree(dev_resv);
    cudaFree(dev_Lv);
    cudaFree(dev_Rv);
    cudaFree(dev_Tv);
    cudaFree(dev_Bv);

    // cudaFree all P related arrays
    cudaFree(dev_p);
    cudaFree(dev_resp);
    cudaFree(dev_Lp);
    cudaFree(dev_Bp);

    // cudaFree all U related arrays
    free(u), free(Lu), free(Ru), free(Tu), free(Bu), free(resu);

    // cudaFree all V related arrays
    free(v), free(Lv), free(Rv), free(Tv), free(Bv), free(resv);

    // cudaFree all P related arrays
    free(p), free(Lp), free(Bp), free(resp);
        
    return(0);
    
} // End main function


// The following are all of the kernels for P, RedU, RedV, BlackU, and BlackV
// The Red and Black kernels respective to U and V may be combined, but they are easier to debug when seperate

__global__ void RedU( double* u, double* Lu, double* Ru, double* Tu, double* Bu, double* p, double* Lp, double* resu )
{
    // Declare local indices for i and j
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    int lj = blockIdx.y * blockDim.y + threadIdx.y;

    if( (lj < dev_N - 1) && (li < dev_N)) // If within NxN-1 for U
    {
        if( (li + lj)%2 == 0 ) // Only update even indices
        {

            // Use the current U and shifted arrays of U to set the residual
            resu[li + lj*dev_N] = (dev_dy/dev_dx)*( Lu[li + lj*dev_N] -2*u[li + lj*dev_N] + Ru[li + lj*dev_N] ) \
                                    + (dev_dx/dev_dy)*( Tu[li + lj*dev_N] -2*u[li + lj*dev_N] + Bu[li + lj*dev_N] ) \
                                    - (dev_dy)*( p[li + lj*dev_N] - Lp[li + lj*dev_N]   );

            u[li + lj*dev_N] = u[li + lj*dev_N] + dev_omega*resu[li + lj*dev_N];
        }
    }
    if ( (lj == dev_N-1) && (li < dev_N) )// Explicitly handle edge case for U
    {
        if( (li + lj)%2 == 0 ) // Only update even indices
        {
            resu[li + lj*dev_N] = 0.0;

            u[li + lj*dev_N] = 0.0;
        }
    }

    __syncthreads();
}

__global__ void RedV( double* v, double* Lv, double* Rv, double* Tv, double* Bv,  double* p, double* Bp,  double* resv )
{
    // Declare local indices for i and j
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    int lj = blockIdx.y * blockDim.y + threadIdx.y;

    if( (lj < dev_N) && (li < dev_N - 1 ) ) // If within NxN-1 of V
    {
        if( (li + lj)%2 == 0 ) // Only update even indices
        {
            // Use the current V and shifted arrays of V to set the residual
            resv[li + lj*dev_N] = (dev_dy/dev_dx)*( Lv[li + lj*dev_N] - 2*v[li + lj*dev_N] + Rv[li + lj*dev_N] ) \
                                    + (dev_dx/dev_dy)*( Tv[li + lj*dev_N] - 2*v[li + lj*dev_N] + Bv[li + lj*dev_N] ) \
                                    - (dev_dx)*( p[li + lj*dev_N] - Bp[li + lj*dev_N] );

            v[li + lj*dev_N] = v[li + lj*dev_N] + dev_omega*resv[li + lj*dev_N];
        }
    }

    if( (lj < dev_N) && (li == dev_N - 1) ) //Explicitly handle edge case for V
    {
        if( (li + lj)%2 == 0 ) // Only update even indices
        {
            resv[li + lj*dev_N] = 0.0;

            v[li + lj*dev_N] = v[li - 1 + lj*dev_N];
        }
    }

    __syncthreads();
}

__global__ void BlackU( double* u, double* Lu, double* Ru, double* Tu, double* Bu, double* p, double* Lp, double* resu  )
{
    // Declare local indices for i and j
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    int lj = blockIdx.y * blockDim.y + threadIdx.y;
    
    if( (lj < dev_N - 1) && (li < dev_N)) // If within NxN-1 of U
    {
        if( (li+lj)%2 == 1 ) // Only uodate odd indices
        {
            // Use the current U  and shifted arrays of U to set the residual
            resu[li + lj*dev_N] = (dev_dy/dev_dx)*( Lu[li + lj*dev_N] -2*u[li + lj*dev_N] + Ru[li + lj*dev_N] ) \                                     + (dev_dx/dev_dy)*( Tu[li + lj*dev_N] -2*u[li + lj*dev_N] + Bu[li + lj*dev_N] ) \
                                    - (dev_dy)*( p[li + lj*dev_N] - Lp[li + lj*dev_N]   );
            
            u[li + lj*dev_N] = u[li + lj*dev_N] + dev_omega*resu[li + lj*dev_N];
        }
    }
    
    if ( (lj == dev_N-1) && (li < dev_N) ) //Explicitly handle edge case of U
    {
        if( (li + lj)%2 == 1 ) // Only update odd indices
        {
            resu[li + lj*dev_N] = 0.0;
            
            u[li + lj*dev_N] = 0.0;
        }
    }
    
    __syncthreads();

}

__global__ void BlackV( double* v, double* Lv, double* Rv, double* Tv, double* Bv,  double* p, double* Bp,  double* resv )
{
    // Declare local indices for i and j
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    int lj = blockIdx.y * blockDim.y + threadIdx.y;

    if( (li < dev_N - 1) && (lj < dev_N ) ) // If within NxN-1 of V
    {
        if( (li + lj)%2 == 1 ) // Only update odd indlices
        {
            // Use the current V and shifted arrays of V to set the residual
            resv[li + lj*dev_N] = (dev_dy/dev_dx)*( Lv[li + lj*dev_N] - 2*v[li + lj*dev_N] + Rv[li + lj*dev_N] ) \
                                    + (dev_dx/dev_dy)*( Tv[li + lj*dev_N] - 2*v[li + lj*dev_N] + Bv[li + lj*dev_N] ) \
                                    - (dev_dx)*( p[li + lj*dev_N] - Bp[li + lj*dev_N] );

            v[li + lj*dev_N] = v[li + lj*dev_N] + dev_omega*resv[li + lj*dev_N];
        }
    }

    if( (li ==  dev_N - 1) && (lj <  dev_N ) ) // explicitly handle edge case of V
    {
        if( (li + lj)%2 == 1 ) // Only update odd indices
        {
            resv[li + lj*dev_N] = 0.0;

            v[li + lj*dev_N] = v[li - 1 + lj*dev_N];
        }
    }

    __syncthreads();
}

__global__ void AllP( double* u, double* Ru, double* v, double* Tv, double* p, double* resp )
{
    // Declare local indices for i and j
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    int lj = blockIdx.y * blockDim.y + threadIdx.y;

    if( (li < dev_N - 1) && (lj < dev_N - 1) ) // If within N-1xN-1 of P
    {
            // Use the current P and shifted arrays of U and V to set the residual
        resp[li + lj*dev_N] = -( Ru[li + lj*dev_N] - u[li + lj*dev_N] ) - (dev_dx/dev_dy)*( Tv[li + lj*dev_N] - v[li + lj*dev_N] );

        p[li + lj*dev_N] = p[li + lj*dev_N] + dev_omega*resp[li + lj*dev_N];
    }


    if( (li < dev_N - 1) &&  (lj == dev_N - 1) )// Explicitly handle edge case of P
    {
        resp[li + lj*dev_N] = 0.0;

        p[li + lj*dev_N] =  0.0;
    }

    if( (li == dev_N - 1) &&  (lj < dev_N - 1) ) // Explicitly handle edge case of P
    {
        resp[li + lj*dev_N] = 0.0;

        p[li + lj*dev_N] =  -p[li - 1 + lj*dev_N];
    }

    __syncthreads();
}









