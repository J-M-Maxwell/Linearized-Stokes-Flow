#include <stdio.h>
#include  <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>


int main(int argc, char* argv[]) {

		int N; // Number of grid points
    double mu; // Fluid viscosity
    int P; // Pressure drop
    double omega; // Relaxation coefficient
    long double tol; // Tolerance for convergence
    int maxiter; // Maximum number of iterations

    N = atoi(argv[1]);
    mu = atof(argv[2]);
    P = atoi(argv[3]);
    omega = atof(argv[4]);
    tol = atof(argv[5]);
    maxiter = atoi(argv[6]);

    double fx = 0;
    double fy = 0;
    double dx = 1./(N-1);
    double dy = dx;

    int i, j, iter;
    
// Declare arrays size at each rank
    double u[N][N];
    double v[N][N-1];
    double p[N-1][N-1];

    // Set initial conditions for U, V, and P
    for(j = 0; j < N-1; ++j)
    {
        for(i =0; i < N-1; ++i)
        {
            u[j][i] = 0;
            v[j][i] = 0;
            p[j][i] = 0;
        }
        u[j][N-1] = 0;
    }
    j = N-1;
    for(i = 0; i < N-1; ++i)
    {
        v[j][i] = 0;
    }

    double res, maxres, residual;
    maxres = 0;
    residual = 1;
    iter = 1;
    while( (iter < maxiter) && (residual > tol) )
    {
        
        // Red U update //
        for( j = 0; j < N-1; ++j)
        {
            for(i = 0; i < N; ++i)
            {
                if( (i+j)%2 == 0)
                {
                    if(j == 0)
                    {
                        if( i == 0)
                        {
                            res = (mu*dy/dx)*( -u[j][i] + u[j][i+1]) + (mu*dx/dy)*(u[j+1][i] - 3*u[j][i] ) - (dy)*(p[j][i] - (2*P - p[j][i]) ) + dx*dy*fy;
                            u[j][i] = u[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                        else
                        {
                            res = (mu*dy/dx)*(u[j][i-1] -2*u[j][i] + u[j][i+1]) + (mu*dx/dy)*(u[j+1][i] - 3*u[j][i] ) - (dy)*(p[j][i] - p[j][i-1] ) + dx*dy*fy;
                            u[j][i] = u[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                    }
                    else if (j == N-2)
                    {
                        if( i == 0)
                        {
                            res = (mu*dy/dx)*( -u[j][i] + u[j][i+1]) + (mu*dx/dy)*( -3*u[j][i] + u[j-1][i]) - (dy)*(p[j][i] - (2*P - p[j][i]) ) + dx*dy*fy;
                            u[j][i] = u[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                        else
                        {
                            res = (mu*dy/dx)*(u[j][i-1] -2*u[j][i] + u[j][i+1]) + (mu*dx/dy)*( -3*u[j][i] + u[j-1][i]) - (dy)*(p[j][i] - p[j][i-1] ) + dx*dy*fy;
                            u[j][i] = u[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                    }
                    else
                    {
                        if( i == 0)
                        {
                            res = (mu*dy/dx)*( -u[j][i] + u[j][i+1]) + (mu*dx/dy)*(u[j+1][i] - 2*u[j][i] + u[j-1][i]) - (dy)*(p[j][i] - (2*P - p[j][i]) ) + dx*dy*fy;
                            u[j][i] = u[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                        else if( i == N-1)
                        {
                            res = (mu*dy/dx)*(u[j][i-1] - u[j][i] ) + (mu*dx/dy)*(u[j+1][i] - 2*u[j][i] + u[j-1][i]) - (dy)*( -2*p[j][i-1] ) + dx*dy*fy;
                            u[j][i] = u[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                        else
                        {
                            res = (mu*dy/dx)*(u[j][i-1] -2*u[j][i] + u[j][i+1]) + (mu*dx/dy)*(u[j+1][i] - 2*u[j][i] + u[j-1][i]) - (dy)*(p[j][i] - p[j][i-1] ) + dx*dy*fy;
                            u[j][i] = u[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                    }
                }
            }
        }
        
        // Red V update //
        for(j = 0; j < N; ++j)
        {
            for( i = 0; i < N-1; ++ i)
            {
                if( (i+j)%2 == 0)
                {
                    if( (j == 0) || (j == N-1) )
                    {
                        res = v[j][i];
                        v[j][i] = 0;
                        if( fabs(res) > maxres)
                            maxres = fabs(res);
                    }
                    else
                    {
                        if( i == 0 )
                        {
                            res = (mu*dy/dx)*( -v[j][i] + v[j][i+1]) + (mu*dx/dy)*(v[j+1][i] -2*v[j][i] + v[j-1][i]) - (dx)*( p[j][i] - p[j-1][i] ) + dx*dy*fx;
                            v[j][i] = v[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                        else if( i == N-2 )
                        {
                            res = (mu*dy/dx)*(v[j][i-1] -v[j][i] ) + (mu*dx/dy)*(v[j+1][i] -2*v[j][i] + v[j-1][i]) - (dx)*(p[j][i] - p[j-1][i] ) + dx*dy*fx;
                            v[j][i] = v[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                        else
                        {
                            res = (mu*dy/dx)*(v[j][i-1] -2*v[j][i] + v[j][i+1]) + (mu*dx/dy)*(v[j+1][i] -2*v[j][i] + v[j-1][i]) - (dx)*(p[j][i] - p[j-1][i]) + dx*dy*fx;
                            v[j][i] = v[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                    }
                }
            }
        }
        
        // Black U update //
        for( j = 0; j < N-1; ++j)
        {
            for(i = 0; i < N; ++i)
            {
                if( (i+j)%2 == 1)
                {
                    if(j == 0)
                    {
                        if( i == N-1)
                        {
                            res = (mu*dy/dx)*(u[j][i-1] -u[j][i] ) + (mu*dx/dy)*(u[j+1][i] - 3*u[j][i] ) - (dy)*( -2*p[j][i-1] ) + dx*dy*fy;
                            u[j][i] = u[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                        else
                        {
                            res = (mu*dy/dx)*(u[j][i-1] -2*u[j][i] + u[j][i+1]) + (mu*dx/dy)*( u[j+1][i] -3*u[j][i] ) - (dy)*(p[j][i] - p[j][i-1]) + dx*dy*fy;
                            u[j][i] = u[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                    }
                    else if (j == N-2)
                    {
                        if( i == N-1)
                        {
                            res = (mu*dy/dx)*(u[j][i-1] -u[j][i] ) + (mu*dx/dy)*( -3*u[j][i] + u[j-1][i]) - (dy)*( -2*p[j][i-1] ) + dx*dy*fy;
                            u[j][i] = u[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                        else
                        {
                            res = (mu*dy/dx)*(u[j][i-1] -2*u[j][i] + u[j][i+1]) + (mu*dx/dy)*( -3*u[j][i] + u[j-1][i]) - (dy)*(p[j][i] - p[j][i-1]) + dx*dy*fy;
                            u[j][i] = u[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                    }
                    else
                    {
                        if( i == 0)
                        {
                            res = (mu*dy/dx)*( -u[j][i] + u[j][i+1]) + (mu*dx/dy)*(u[j+1][i] - 2*u[j][i] + u[j-1][i]) - (dy)*(p[j][i] - (2*P - p[j][i]) ) + dx*dy*fy;
                            u[j][i] = u[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                        else if( i == N-1)
                        {
                            res = (mu*dy/dx)*(u[j][i-1] -u[j][i] ) + (mu*dx/dy)*(u[j+1][i] - 2*u[j][i] + u[j-1][i]) - (dy)*( -2*p[j][i-1] ) + dx*dy*fy;
                            u[j][i] = u[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                        else
                        {
                            res = (mu*dy/dx)*(u[j][i-1] -2*u[j][i] + u[j][i+1]) + (mu*dx/dy)*(u[j+1][i] - 2*u[j][i] + u[j-1][i]) - (dy)*(p[j][i] - p[j][i-1]) + dx*dy*fy;
                            u[j][i] = u[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                    }
                }
            }
        }
        
        //Black V update //
        for(j = 0; j < N; ++j)
        {
            for( i = 0; i < N-1; ++ i)
            {
                if( (i+j)%2 == 1)
                {
                    if( (j == 0) || (j == N-1) )
                    {
                        res = v[j][i];
                        v[j][i] = 0;
                        if( fabs(res) > maxres)
                            maxres = fabs(res);
                    }
                    else
                    {
                        if( i == 0 )
                        {
                            res = (mu*dy/dx)*( -v[j][i] + v[j][i+1]) + (mu*dx/dy)*(v[j+1][i] -2*v[j][i] + v[j-1][i]) - (dx)*(p[j][i] - p[j-1][i]) + dx*dy*fx;
                            v[j][i] = v[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                        else if( i == N-2 )
                        {
                            res = (mu*dy/dx)*(v[j][i-1] -v[j][i] ) + (mu*dx/dy)*(v[j+1][i] -2*v[j][i] + v[j-1][i]) - (dx)*(p[j][i] - p[j-1][i]) + dx*dy*fx;
                            v[j][i] = v[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                        else
                        {
                            res = (mu*dy/dx)*(v[j][i-1] -2*v[j][i] + v[j][i+1]) + (mu*dx/dy)*(v[j+1][i] -2*v[j][i] + v[j-1][i]) - (dx)*(p[j][i] - p[j-1][i]) + dx*dy*fx;
                            v[j][i] = v[j][i] + omega*res;
                            if( fabs(res) > maxres)
                                maxres = fabs(res);
                        }
                    }
                }
            }
        }
        
        // Update all P grid points //
        for( j = 0; j < N-1; ++j)
        {
            for(i = 0; i < N-1; ++i)
            {
                res = -(u[j][i+1] - u[j][i]) - (dx/dy)*(v[j+1][i] - v[j][i]);
                p[j][i] = p[j][i] + omega*res;
                if( fabs(res) > maxres)
                    maxres = fabs(res);
            }
        }
        
        residual = maxres;
        maxres = 0;
        iter += 1;
    }
 
    double ufin[N*(N-1)];
    double vfin[N*(N-1)];
    double pfin[(N-1)*(N-1)];
    
    for(j = 0; j < N-1; ++j)
    {
        for( i = 0; i < N-1; ++i)
        {
            ufin[j*N + i] = u[j][i];
            vfin[j*(N-1) + i] = v[j][i];
            pfin[j*(N-1) + i] = p[j][i];
        }
        ufin[j*N + N-1] = u[j][N-1];
    }
    for( i = 0; i < N-1; ++i)
    {
        vfin[(N-1)*N + i] = v[N-1][i];
    }
    
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
    
    
    return(0);
    
}
