import numpy as np
from scipy.spatial.transform import Rotation as R

rot = np.zeros((3,3))
roll = np.zeros((3,3))
yaw = np.zeros((3,3))
pitch = np.zeros((3,3))

q1 = [ 0.998629, 0.000000, -0.052337, 0.000000]
q2 = [ 0.999391, 0.000000, -0.034901, 0.000000]
r = R.from_quat(q2 - q1)
r.apply([1,1,1])

r.
