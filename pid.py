# Import necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import neuromancer.psl as psl
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.plot import pltCL

# Ground truth system model
sys = psl.systems['LinearSimpleSingleZone']()

# Problem dimensions
nx = sys.nx           # Number of states
nu = sys.nu           # Number of control inputs
nd = sys.nD           # Number of disturbances
nd_obs = sys.nD_obs   # Number of observable disturbances
ny = sys.ny           # Number of controlled outputs
nref = ny             # Number of references

# Control action bounds
umin = torch.tensor(sys.umin)
umax = torch.tensor(sys.umax)

# Function to generate training data
def get_data(sys, nsteps, n_samples, xmin_range, batch_size, name="train"):
    # Sampled references for training the policy
    batched_xmin = xmin_range.sample((n_samples, 1, nref)).repeat(1, nsteps + 1, 1)
    batched_xmax = batched_xmin + 2.

    # Sampled disturbance trajectories from the simulation model
    batched_dist = torch.stack([torch.tensor(sys.get_D(nsteps)) for _ in range(n_samples)])

    # Sampled initial conditions
    batched_x0 = torch.stack([torch.tensor(sys.get_x0()).unsqueeze(0) for _ in range(n_samples)])

    # Create a dataset
    data = DictDataset(
        {"x": batched_x0,            # Initial states
         "y": batched_x0[:, :, [-1]], # Output values corresponding to the last state
         "ymin": batched_xmin,        # Minimum comfort bounds
         "ymax": batched_xmax,        # Maximum comfort bounds
         "d": batched_dist},          # Disturbance trajectories
        name=name,
    )

    return DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, shuffle=False)

# Define parameters for data generation
nsteps = 100  # Prediction horizon
n_samples = 1000    # Number of sampled scenarios
batch_size = 100

# Range for lower comfort bound
xmin_range = torch.distributions.Uniform(18., 22.)

# Generate training and validation data
train_loader, dev_loader = [
    get_data(sys, nsteps, n_samples, xmin_range, batch_size, name=name)
    for name in ("train", "dev")
]

# Extract exact state space model matrices
A = torch.tensor(sys.A)
B = torch.tensor(sys.Beta)
C = torch.tensor(sys.C)
E = torch.tensor(sys.E)

# State-space model of the building dynamics
#   x_k+1 =  A x_k + B u_k + E d_k
xnext = lambda x, u, d: x @ A.T + u @ B.T + d @ E.T
state_model = Node(xnext, ['x', 'u', 'd'], ['x'], name='SSM')

#   y_k = C x_k
ynext = lambda x: x @ C.T
output_model = Node(ynext, ['x'], ['y'], name='y=Cx')

# Partially observable disturbance model
dist_model = lambda d: d[:, sys.d_idx]
dist_obs = Node(dist_model, ['d'], ['d_obs'], name='dist_obs')

# Define a simple PID controller class
class PIDController(nn.Module):
    def __init__(self, Kp, Ki, Kd, umin, umax):
        super(PIDController, self).__init__()
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.umin = umin
        self.umax = umax
        self.integral = 0.0
        self.prev_error = 0.0

    def forward(self, y, ymin, ymax, d_obs):
        # Calculate the error between the desired and actual values
        error = (ymax + ymin) / 2.0 - y
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        # Calculate the control action using PID formula
        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        u = torch.clamp(u, self.umin, self.umax)  # Ensure the control action is within bounds

        return u

# PID controller parameters
Kp = 1.0
Ki = 0.1
Kd = 0.01

# Instantiate PID controller
pid_controller = PIDController(Kp, Ki, Kd, umin, umax)
policy = Node(pid_controller, ['y', 'ymin', 'ymax', 'd_obs'], ['u'], name='policy')

# Closed-loop system model
cl_system = System([dist_obs, policy, state_model, output_model],
                   nsteps=nsteps,
                   name='cl_system')
cl_system.show()

# Variables for constraints and objectives
y = variable('y')
u = variable('u')
ymin = variable('ymin')
ymax = variable('ymax')

# Objectives for the control problem
action_loss = 0.01 * (u == 0.0)  # Energy minimization (encourages minimal control action)
du_loss = 0.1 * (u[:, :-1, :] - u[:, 1:, :] == 0.0)  # Delta u minimization to prevent aggressive changes in control actions

# Thermal comfort constraints
state_lower_bound_penalty = 50. * (y > ymin)  # Penalty for violating lower bound on temperature
state_upper_bound_penalty = 50. * (y < ymax)  # Penalty for violating upper bound on temperature

# Objectives and constraints names for nicer plot
action_loss.name = 'action_loss'
du_loss.name = 'du_loss'
state_lower_bound_penalty.name = 'x_min'
state_upper_bound_penalty.name = 'x_max'

# List of constraints and objectives
objectives = [action_loss, du_loss]
constraints = [state_lower_bound_penalty, state_upper_bound_penalty]

# Data (x_k, r_k) -> Parameters (xi_k) -> Policy (u_k) -> Dynamics (x_k+1)
nodes = [cl_system]
# Create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)
# Construct constrained optimization problem
problem = Problem(nodes, loss)
# Plot computational graph
problem.show()

# Define test parameters
nsteps_test = 2000

# Generate reference
np_refs = psl.signals.step(nsteps_test + 1, 1, min=18., max=22., randsteps=5)
ymin_val = torch.tensor(np_refs, dtype=torch.float32).reshape(1, nsteps_test + 1, 1)
ymax_val = ymin_val + 2.0
# Generate disturbance signal
torch_dist = torch.tensor(sys.get_D(nsteps_test + 1)).unsqueeze(0)
# Initial data for closed loop simulation
x0 = torch.tensor(sys.get_x0()).reshape(1, 1, nx)
data = {'x': x0,
        'y': x0[:, :, [-1]],
        'ymin': ymin_val,
        'ymax': ymax_val,
        'd': torch_dist}
cl_system.nsteps = nsteps_test
# Perform closed-loop simulation
trajectories = cl_system(data)

# Constraints bounds
Umin = umin * np.ones([nsteps_test, nu])
Umax = umax * np.ones([nsteps_test, nu])
Ymin = trajectories['ymin'].detach().reshape(nsteps_test + 1, nref)
Ymax = trajectories['ymax'].detach().reshape(nsteps_test + 1, nref)

# Plot closed loop trajectories
pltCL(Y=trajectories['y'].detach().reshape(nsteps_test + 1, ny),
      R=Ymax,
      X=trajectories['x'].detach().reshape(nsteps_test + 1, nx),
      D=trajectories['d'].detach().reshape(nsteps_test + 1, nd),
      U=trajectories['u'].detach().reshape(nsteps_test, nu),
      Umin=Umin, Umax=Umax, Ymin=Ymin, Ymax=Ymax)

