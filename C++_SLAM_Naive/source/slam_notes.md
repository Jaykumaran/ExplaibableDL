

## Pose matrix in SE(3)

👉 **Always do rotation first then translation, not the other way**

Mathematically a camera “pose” is an element of the Lie group SE(3), which you can write in homogeneous-coordinates as a 4×4 matrix:

$$
T \;=\;
\begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix},
$$

where  
- \(R \in SO(3)\) is a \(3\times3\) rotation matrix,  
- \(t \in \mathbb{R}^3\) is a translation vector,  
- and the bottom row \([0\;0\;0\;1]\) makes it a homogeneous transform.

---

When you linearize your SLAM cost around a current pose estimate, you work in the tangent space \(\mathfrak{se}(3)\), which is a 6-dimensional vector space. In a Gauss–Newton or Levenberg–Marquardt step you end up solving a normal-equation block for each pose:

$$
H\,\Delta \xi = b,
$$

where  
- \(\Delta \xi \in \mathbb{R}^6\) is the 6-vector of incremental pose parameters,  
- \(H = \sum_i J_i^\top\,\Omega_i\,J_i \in \mathbb{R}^{6\times6}\) is the Hessian (information) block for that pose,  
- \(J_i\) is the Jacobian of the \(i\)th residual w.r.t.\ the 6 pose parameters,  
- \(\Omega_i\) is the measurement information matrix.

---

In g2o’s code:

```cpp
using PoseMatrixType = Eigen::Matrix<double,6,6>;



Pose = 6D (3 rot + 3 trans).

Each pose–pose block in the Hessian is 6×6.

You build those blocks from the Jacobians  
`J_pose^T J_pose`.

Use a Schur complement to eliminate 3D points, leaving a smaller 6×6 system per pose to solve.

Convert the 6×1 solution back into an SE3 update via the exponential map.
