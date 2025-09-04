import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Quaternion utility functions
def quat_mult(q, r):
    """Quaternion multiplication q ⊗ r"""
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def quat_normalize(q):
    return q / np.linalg.norm(q)

def rotate_points(points, q):
    """Rotate 3D points using quaternion q"""
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    rotated = []
    for p in points:
        p_quat = np.array([0, *p])
        rotated_p = quat_mult(quat_mult(q, p_quat), q_conj)
        rotated.append(rotated_p[1:])
    return np.array(rotated)

# Parameters
axis = np.array([0, 0, 1], dtype=float)   # rotation axis (z-axis)
omega = 1.0  # angular velocity (rad/s)
dt = 0.05    # timestep

# Initial quaternion (identity rotation)
q = np.array([1, 0, 0, 0], dtype=float)

# Define cube vertices (centered at origin)
cube_vertices = np.array([
    [-0.5, -0.5, -0.5],
    [ 0.5, -0.5, -0.5],
    [ 0.5,  0.5, -0.5],
    [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5,  0.5],
    [-0.5,  0.5,  0.5]
])

# Faces of the cube (list of lists of vertex indices)
faces = [
    [0, 1, 2, 3],  # bottom
    [4, 5, 6, 7],  # top
    [0, 1, 5, 4],  # front
    [2, 3, 7, 6],  # back
    [1, 2, 6, 5],  # right
    [0, 3, 7, 4]   # left
]

# Colors for each face
face_colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]

# Setup figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

poly_collection = []  # store cube faces
text_display = None   # store text overlay

def init():
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    # Draw coordinate system
    ax.quiver(0, 0, 0, 1, 0, 0, color="r")
    ax.quiver(0, 0, 0, 0, 1, 0, color="g")
    ax.quiver(0, 0, 0, 0, 0, 1, color="b")
    return []

def update(frame):
    global q, axis, omega, poly_collection, text_display

    # Remove old cube faces
    for poly in poly_collection:
        poly.remove()
    poly_collection = []

    # Remove old text
    if text_display is not None:
        text_display.remove()

    # Quaternion time derivative: dq/dt = 0.5 * q ⊗ omega_q
    if np.linalg.norm(axis) > 1e-8:
        axis_norm = axis / np.linalg.norm(axis)
    else:
        axis_norm = np.array([0,0,1])
    omega_q = np.array([0, *(omega * axis_norm)])
    dqdt = 0.5 * quat_mult(q, omega_q)
    q = quat_normalize(q + dqdt * dt)

    # Rotate cube
    rotated_vertices = rotate_points(cube_vertices, q)

    # Draw cube faces
    for face, color in zip(faces, face_colors):
        square = [rotated_vertices[i] for i in face]
        poly = Poly3DCollection([square], facecolors=color, alpha=0.7)
        ax.add_collection3d(poly)
        poly_collection.append(poly)

    # Add text overlay
    quat_str = f"[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]"
    axis_str = f"[{axis_norm[0]:.2f}, {axis_norm[1]:.2f}, {axis_norm[2]:.2f}]"
    text_display = ax.text2D(
        0.02, 0.95,
        f"Axis: {axis_str}\nSpeed: {omega:.2f} rad/s\nQuat: {quat_str}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace"
    )

    return poly_collection + [text_display]


# Key press handling
def on_key(event):
    global axis, omega, q
    step = 0.2
    if event.key == "left":
        axis[0] -= step
    elif event.key == "right":
        axis[0] += step
    elif event.key == "up":
        axis[1] += step
    elif event.key == "down":
        axis[1] -= step
    elif event.key in ["u", "pageup"]:
        axis[2] += step
    elif event.key in ["d", "pagedown"]:
        axis[2] -= step
    elif event.key == "+":
        omega += 0.2
    elif event.key == "-":
        omega -= 0.2
    elif event.key == "r":
        q[:] = [1,0,0,0]
        axis[:] = [0,0,1]
        omega = 1.0
    print(f"Axis={axis}, omega={omega:.2f}")

fig.canvas.mpl_connect("key_press_event", on_key)

ani = FuncAnimation(fig, update, init_func=init, frames=200, interval=50, blit=False)
plt.show()