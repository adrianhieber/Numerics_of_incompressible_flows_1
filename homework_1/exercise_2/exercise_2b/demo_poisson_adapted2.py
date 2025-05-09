
from dolfin import *

# Create mesh and define function space
mesh = RectangleMesh(Point(0, 0), Point(1, 1), 32, 32,'left')
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x,y = 0 or x,y = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

# Define boundary condition
u_D = Expression("1 + x[0]*x[0] + 2*x[1]*x[1]", degree=2)
bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("-6", degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx #+ g*v*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution in pvd-vtu format
file = File("poisson.pvd")
file << u

# Plot solution
import matplotlib.pyplot as plt
plot(u)
plt.show()
