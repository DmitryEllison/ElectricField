from matplotlib import pyplot
import numpy as np

def norm(x):
    return np.sqrt(np.sum(np.array(x)**2, axis = -1))

def angle (x0, x1, x2):
    a,b = x1 - x0, x1 - x2
    return np.arccos(np.dot(a,b)/(norm(a)*norm(b)))


def perp_point_line(x0, x1, x2):
    #module of shortest distance between the point and the line
    #fabs возвращает модуль векторного умножения векторов
    # (расстояние от точки до одного конца),
            #(расстояние от точки до другого конца)
    return np.fabs(np.cross(x0-x1, x0-x2))/norm(x2-x1)

def is_left(x0, x1, x2):
    #true if x0 if on the left of the line x1-x2
    matrix = np.array([x1-x0, x2-x0])
    if len(x0.shape) == 2:
        matrix = matrix.transpose((1,2,0))
    return np.linalg.det(matrix) > 0

class ElectricField:

    dt0 = 0.01  # The time step for integrations

    def __init__(self, charges):
        #Initializes the field given 'charges'.
        self.charges = charges

    def vector(self, x):
        #Returns the field vector.
        return np.sum([charge.E(x) for charge in self.charges], axis=0)

    def magnitude(self, x):
        #Returns the magnitude of the field vector.
        return norm(self.vector(x))

    def angle(self, x):
        #Returns the field vector's angle from the x-axis (in radians).
        return np.arctan2(*(self.vector(x).T[::-1])) # arctan2 gets quadrant right


class Line:
    # a projection of a charged line
    R = 0.01
    
    def __init__(self, q, x1, x2):
        #creates an instance of a line with charge q
                        #between points x1 and x2
        self.q, self.x1, self.x2 = q, np.array(x1), np.array(x2)
        
    def get_lambda(self):
        #returns lambda
        return self.q / norm(self.x2 - self.x1)
    lam = property(get_lambda)
    
    def E(self, x):
        #electric field vector
        x = np.array(x)
        x1, x2, lam = self.x1, self.x2, self.lam
        theta1, theta2 = angle(x,x1,x2), np.pi - angle(x, x2, x1)
        a = perp_point_line(x, x1, x2)
        r1, r2 = norm(x - x1), norm(x - x2)
        sign = np.where(is_left(x, x1, x2), 1, -1)
        
        E_parallel = lam*(1/r2 - 1/r1)
        E_perpendicular = -sign*lam*(np.cos(theta2 - theta1)) \
                                       / np.where(a == 0, np.infty, a)
        dx = x2 - x1    
        if len(x.shape) == 2:
            E_parallel = E_parallel[::, np.newaxis]
            E_perpendicular = E_perpendicular[::, np.newaxis]
        return E_perpendicular *(np.array([-dx[1], dx[0]])/norm(dx)) \
                                      +  E_parallel * (dx/norm(dx))
    def is_close(self, x):
        #Returns True if x is close to the charge.

        theta1 = angle(x, self.x1, self.x2)
        theta2 = angle(x, self.x2, self.x1)

        if theta1 < np.radians(90) and theta2 < np.radians(90):
            return perp_point_line(x, self.x1, self.x2) < self.R
        return np.min([norm(self.x1-x), norm(self.x2-x)], axis=0) < self.R
        
    def plot(self):
        #Plots the charge.
        color = 'b' if self.q < 0 else 'r' if self.q > 0 else 'k'
        x, y = zip(self.x1, self.x2)
        width = 5*(np.sqrt(np.fabs(self.lam))/2 + 1)
        pyplot.plot(x, y, color, linewidth=width)

class Point:

    R = 0.01  # The effective radius of the charge

    def __init__(self, q, x):
        #Initializes the quantity of charge 'q' and position vector 'x'.
        self.q, self.x = q, np.array(x)

    def E(self, x):  
        #Electric field vector.
        if self.q == 0:
            return 0
        dx = x-self.x
        return (self.q*dx.T/np.sum(dx**2, axis=-1)**1.5).T

    def is_close(self, x):
        #Returns True if x is close to the charge; false otherwise.
        return norm(x-self.x) < self.R

    def plot(self):
        #Plots the charge.
        color = 'b' if self.q < 0 else 'r' if self.q > 0 else 'k'
        r = 0.1*(np.sqrt(np.fabs(self.q))/2 + 1)
        circle = pyplot.Circle(self.x, r, color=color, zorder=10)
        pyplot.gca().add_artist(circle)

def finalize_plot():
    #Finalizes the plot.
    ax = pyplot.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    pyplot.xlim(XMIN/ZOOM+XOFFSET, XMAX/ZOOM+XOFFSET)
    pyplot.ylim(YMIN/ZOOM, YMAX/ZOOM)
    pyplot.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    
    
XMIN, XMAX = -40, 40
YMIN, YMAX = -30, 30
ZOOM = 9
XOFFSET = 0

# Set up the charges, electric field, and potential
charges = [Point(-50, [3, 2]), 
           Point(5, [-3, -2]),
           Point(5, [-3, 2]),
           Point(5, [3, -2]),
           Point(1, [0, 0])]
field = ElectricField(charges)

# Create the vector grid
x, y = np.meshgrid(np.linspace(XMIN/ZOOM+XOFFSET, XMAX/ZOOM+XOFFSET, 41),
                      np.linspace(YMIN/ZOOM, YMAX/ZOOM, 31))
u, v = np.zeros_like(x), np.zeros_like(y)
n, m = x.shape
for i in range(n):
    for j in range(m):
        if any(charge.is_close([x[i, j], y[i, j]]) for charge in charges):
            u[i, j] = v[i, j] = None
        else:
            mag = field.magnitude([x[i,j], y[i,j]])**(1/5)
            a = field.angle([x[i,j], y[i,j]])
            u[i, j], v[i, j] = mag*np.cos(a), mag*np.sin(a)

# Field vectors
fig = pyplot.figure(figsize=(6, 4.5))
cmap = pyplot.cm.get_cmap('plasma')
pyplot.quiver(x, y, u, v, pivot='mid', cmap=cmap, scale=35)
for charge in charges:
    charge.plot()
    
finalize_plot()

pyplot.show()
