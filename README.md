# Serial-Linearized-Stokes-Flow
Numerical method for solving the Stokes Flow problem, a linearization of the Navier-Stokes equations for modeling fluid flow.


Program Solves Stokes equation on using a marker-and-cell (MAC) method using successive over relaxation and alternating Red-Black ordering, for the U, V, and P. 
(U, V) are the fluid velocity and P is the fluid pressure.  
 
 The general algorithm follows the steps below.
 
A) Initiates the relevent initial conditions for U, V, and P
 
B) The program updates the grid points of U, V, and P in the following order.
       1) Update red points of U
       2) Update red points of V
       3) Update black points of U
       4) Update black points of V
       5) Update all point of P
       
C) Calculates the maximum residual and compares to error tolerance for convergence and repeats (B) until error tolerance is met.

D) Gathers final solution across all ranks and writes to file.
