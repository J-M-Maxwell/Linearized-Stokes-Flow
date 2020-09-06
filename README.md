Numerical method for solving the Stokes Flow problem, a linearization of the Navier-Stokes equations for modeling fluid flow.

Programs solve the Linearized Stokes Flow equations using marker-and-cell (MAC) successive over relaxation and alternating Red-Black ordering, for the U, V, and P equations. 
(U, V) are the fluid velocity and P is the fluid pressure.  
 
 
 
 
 
 inputs: argc should be 6
        argv[1]: N - Size of grid
        argv[2]: mu
        argv[3]: P - Pressure drop
        argv[4]: omega - relaxation coefficient
        argv[5]: tol - tolerance for convergence
        argv[6]: max_iter - max iterations allowed
outputs:
        Final solutions for U, V, and P saved to
        StokesU.out, StokesV.out, and StokesP.out
       
       
       
 
 The general algorithm follows the steps below.
 
A) Initiates the relevent initial conditions for U, V, and P
 
B) The program updates the grid points of U, V, and P in the following order.
       1) Update red points of U
       2) Update red points of V
       3) Update black points of U
       4) Update black points of V
       5) Update all points of P
       
C) Calculates the maximum residual and compares to error tolerance for convergence and repeats (B) until error tolerance is met

D) Gathers final solution writes to file
       1) If program is in parrallel using MPI, the solution is gather accross all ranks 
       2) If program is using CUDA GPU programming, the solution is gathered accross all instances



~~ If program is using CUDA ~~

The stencils for the Red and Black update of U and V at grid point (i, j) relies on the neighboring indices (i+1, j), (i-1, j), (i, j+1), and (i, j-1). Instead of explicitly using the neighboring grid points in the kernel for each update, the kernel takes shifted arrays for U, V, and P as arguements.
For example a kernel for the U update takes Lu, Ru, Tu, and Bu as arguements where these arrays are copies of U shifted one to the left, right, top, and bottom respectively.

       
       
       
       
       
       
       
