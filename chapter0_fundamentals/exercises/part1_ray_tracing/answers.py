#%%
import os
import sys
import torch as t
from torch import Tensor
import einops
from ipywidgets import interact
import plotly.express as px
from ipywidgets import interact
from pathlib import Path
from IPython.display import display
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
assert exercises_dir.name == "exercises", f"This file should be run inside 'exercises/part1_ray_tracing', not '{section_dir}'"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle
import part1_ray_tracing.tests as tests

MAIN = __name__ == "__main__"

#%%
def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    '''
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    '''
    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    rays[:, 1, 0] = 1
    return rays

rays1d = make_rays_1d(9, 10.0)

if MAIN:
    fig = render_lines_with_plotly(rays1d)
# %%
if MAIN:
    fig = setup_widget_fig_ray()
    display(fig)

@interact
def response(seed=(0, 10, 1), v=(-2.0, 2.0, 0.01)):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(-2), P(2))
    with fig.batch_update(): 
        fig.data[0].update({"x": x, "y": y}) 
        fig.data[1].update({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}) 
        fig.data[2].update({"x": [P(v)[0]], "y": [P(v)[1]]})
# %%
segments = t.tensor([
    [[1.0, -12.0, 0.0], [1, -6.0, 0.0]], 
    [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], 
    [[2, 12.0, 0.0], [2, 21.0, 0.0]]
])

render_lines_with_plotly(rays1d, segments)
# %%
def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    
    ray = ray[..., :2]
    segment = segment[..., :2]

    O, D = ray
    L_1, L_2 = segment

    mat = t.stack([D, L_1 - L_2], dim=-1)
    vec = L_1 - O

    try:
        sol = t.linalg.solve(mat, vec)
    except:
        return False

    u = sol[0].item()
    v = sol[1].item()
    return (u >= 0.0) and (v >= 0.0) and (v <= 1.0)
# %%
@jaxtyped
@typeguard.typechecked
def intersect_ray_1d(ray: Float[Tensor, "points=2 dim=3"], segment: Float[Tensor, "points=2 dim=3"]) -> bool:
    ray = ray[..., :2]
    segment = segment[..., :2]

    O, D = ray
    L_1, L_2 = segment

    mat = t.stack([D, L_1 - L_2], dim=-1)
    vec = L_1 - O

    try:
        sol = t.linalg.solve(mat, vec)
    except:
        return False

    u = sol[0].item()
    v = sol[1].item()
    return (u >= 0.0) and (v >= 0.0) and (v <= 1.0)
# %%
def intersect_rays_1d(rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if it intersects any segment.
    '''
    NR = rays.size(0)
    NS = segments.size(0)

    rays = rays[..., :2]
    segments = segments[..., :2]

    rays = einops.repeat(rays, "nrays p d -> nrays nsegments p d", nsegments=NS)
    segments = einops.repeat(segments, "nsegments p d -> nrays nsegments p d", nrays=NR)

    O = rays[:, :, 0]
    D = rays[:, :, 1]
    assert O.shape == (NR, NS, 2)

    L_1 = segments[:, :, 0]
    L_2 = segments[:, :, 1]
    assert L_1.shape == (NR, NS, 2)

    mat = t.stack([D, L_1 - L_2], dim=-1)
    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    assert is_singular.shape == (NR, NS)
    mat[is_singular] = t.eye(2)

    vec = L_1 - O

    sol = t.linalg.solve(mat, vec)
    u = sol[..., 0]
    v = sol[..., 1]

    return ((u >= 0) & (v >= 0) & (v <= 1) & ~is_singular).any(dim=-1)

if MAIN:
    tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)
# %%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[t.Tensor, "nrays 2 3"]:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''
    n_pixels = num_pixels_y * num_pixels_z
    ygrid = t.linspace(-y_limit, y_limit, num_pixels_y)
    zgrid = t.linspace(-z_limit, z_limit, num_pixels_z)
    rays = t.zeros((n_pixels, 2, 3), dtype=t.float32)
    rays[:, 1, 0] = 1
    rays[:, 1, 1] = einops.repeat(ygrid, "y -> (y z)", z=num_pixels_z)
    rays[:, 1, 2] = einops.repeat(zgrid, "z -> (y z)", y=num_pixels_y)
    return rays


if MAIN:
    rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
    render_lines_with_plotly(rays_2d)
# %%
if MAIN:
    one_triangle = t.tensor([[0, 0, 0], [3, 0.5, 0], [2, 3, 0]])
    A, B, C = one_triangle
    x, y, z = one_triangle.T

    fig = setup_widget_fig_triangle(x, y, z)

@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
def response(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.data[2].update({"x": [P[0]], "y": [P[1]]})


if MAIN:
    display(fig)
# %%
Point = Float[Tensor, "points=3"]

@jaxtyped
@typeguard.typechecked
def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    '''
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    '''
    s, u, v = t.linalg.solve(
        t.stack([-D, B - A, C - A], dim=1), 
        O - A
    )
    return ((u >= 0) & (v >= 0) & (u + v <= 1)).item()


if MAIN:
    tests.test_triangle_ray_intersects(triangle_ray_intersects)
# %%
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    NR = rays.size(0)
    A, B, C = einops.repeat(triangle, "trianglePoints dims -> trianglePoints nrays dims", nrays=NR)
    assert A.shape == (NR, 3)

    O, D = rays.unbind(dim=1)
    assert O.shape == (NR, 3)

    mat: Float[Tensor, "nrays 3 3"] = t.stack([-D, B - A, C - A], dim=-1)
    dets: Float[Tensor, "nrays"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol: Float[Tensor, "nrays 3"] = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)


if MAIN:
    A = t.tensor([1, 0.0, -0.5])
    B = t.tensor([1, -0.5, 0.0])
    C = t.tensor([1, 0.5, 0.5])
    num_pixels_y = num_pixels_z = 15
    y_limit = z_limit = 0.5

    # Plot triangle & rays
    test_triangle = t.stack([A, B, C], dim=0)
    rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
    render_lines_with_plotly(rays2d, triangle_lines)

    # Calculate and display intersections
    intersects = raytrace_triangle(rays2d, test_triangle)
    img = intersects.reshape(num_pixels_y, num_pixels_z).int()
    imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")
# %%
if MAIN:
    with open(section_dir / "pikachu.pt", "rb") as f:
        triangles = t.load(f)
# %%
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''
    NR = rays.size(0)
    NT = triangles.size(0)

    triangles = einops.repeat(triangles, "ntriangles trianglePoints dims -> trianglePoints nrays ntriangles dims", nrays=NR)
    rays = einops.repeat(rays, "nrays rayPoints dims -> rayPoints nrays ntriangles dims", ntriangles=NT)
    A, B, C = triangles
    O, D = rays
    assert A.shape == (NR, NT, 3)
    assert O.shape == (NR, NT, 3)

    mat: Float[Tensor, "nrays ntriangles 3 3"] = t.stack([-D, B - A, C - A], dim=-1)
    dets: Float[Tensor, "nrays ntriangles"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec: Float[Tensor, "nrays ntriangles 3"] = O - A

    sol: Float[Tensor, "nrays ntriangles 3"] = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    intersects = ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)
    s[~intersects] = t.inf

    return s.min(dim=-1).values


if MAIN:
    num_pixels_y = 120
    num_pixels_z = 120
    y_limit = z_limit = 1

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0] = t.tensor([-2, 0.0, 0.0])
    dists = raytrace_mesh(rays, triangles)
    intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
    dists_square = dists.view(num_pixels_y, num_pixels_z)
    img = t.stack([intersects, dists_square], dim=0)

    fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
    fig.update_layout(coloraxis_showscale=False)
    for i, text in enumerate(["Intersects", "Distance"]): 
        fig.layout.annotations[i]['text'] = text
    fig.show()
# %%
from typing import Callable
from tqdm import tqdm

def raytrace_mesh_video(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
    rotation_matrix: Callable[[Float[Tensor, "nrays rayPoints=2 dims=3"]], Float[Tensor, "3 3"]],
    num_frames: int,
) -> Float[Tensor, "nframes nrays"]:
    result = []
    theta = t.tensor(2 * t.pi) / num_frames
    R = rotation_matrix(theta)
    for theta in tqdm(range(num_frames)):
        triangles = triangles @ R
        result.append(raytrace_mesh(rays, triangles))
    return t.stack(result, dim=0)

if MAIN:
    num_pixels_y = 200
    num_pixels_z = 200
    y_limit = z_limit = 1
    num_frames = 50

    rotation_matrix = lambda theta: t.tensor([
        [t.cos(theta), 0.0, t.sin(theta)],
        [0.0, 1.0, 0.0],
        [-t.sin(theta), 0.0, t.cos(theta)],
    ])

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0] = t.tensor([-2, 0.0, 0.0])
    dists = raytrace_mesh_video(rays, triangles, rotation_matrix, num_frames)
    dists_square = dists.view(num_frames, num_pixels_y, num_pixels_z)

    fig = px.imshow(dists_square, animation_frame=0, origin="lower", color_continuous_scale="viridis_r")
    # zmin=0, zmax=2, color_continuous_scale="Brwnyl"
    fig.update_layout(coloraxis_showscale=False)
    fig.show()
# %%
from einops import repeat, reduce, rearrange

def make_rays_2d_origin(
    num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float, origin: t.Tensor
) -> t.Tensor:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.
    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''
    rays = t.zeros((num_pixels_y, num_pixels_z, 2, 3))
    rays[:, :, 1, 0] = 1
    rays[:, :, 1, 1] = repeat(
        t.arange(num_pixels_y) * 2.0 * y_limit / (num_pixels_y - 1) - y_limit, "y -> y z", z=num_pixels_z
    )
    rays[:, :, 1, 2] = repeat(
        t.arange(num_pixels_z) * 2.0 * z_limit / (num_pixels_z - 1) - z_limit, "z -> y z", y=num_pixels_y
    )
    rays[:, :, 0, :] = origin
    return rearrange(rays, "y z n d -> (y z) n d", n=2, d=3)

def raytrace_mesh_gpu(triangles: t.Tensor, rays: t.Tensor) -> t.Tensor:
    '''For each ray, return the distance to the closest intersecting triangle, or infinity.
    triangles: shape (n_triangles, n_points=3, n_dims=3)
    rays: shape (n_pixels, n_points=2, n_dims=3)
    return: shape (n_pixels, )
    '''
    n_triangles = triangles.size(0)
    n_pixels = rays.size(0)
    device = "cuda"
    matrices = t.zeros((n_pixels, n_triangles, 3, 3)).to(device)
    rays_gpu = rays.to(device)
    matrices[:, :, :, 0] = repeat(rays_gpu[:, 0] - rays_gpu[:, 1], "r d -> r t d", t=n_triangles)
    triangles_gpu = triangles.to(device)
    matrices[:, :, :, 1] = repeat(triangles_gpu[:, 1] - triangles_gpu[:, 0], "t d -> r t d", r=n_pixels)
    matrices[:, :, :, 2] = repeat(triangles_gpu[:, 2] - triangles_gpu[:, 0], "t d -> r t d", r=n_pixels)
    bs = repeat(rays_gpu[:, 0], "r d -> r t d", t=n_triangles) - repeat(
        triangles_gpu[:, 0], "t d -> r t d", r=n_pixels
    )
    mask = t.linalg.det(matrices) != 0
    distances = t.full((n_pixels, n_triangles), float("inf")).to(device)
    solns = t.linalg.solve(matrices[mask], bs[mask])
    distances[mask] = t.where(
        (solns[:, 0] >= 0) & (solns[:, 1] >= 0) & (solns[:, 2] >= 0) & (solns[:, 1] + solns[:, 2] <= 1),
        solns[:, 0],
        t.tensor(float("inf")).to(device),
    )
    return reduce(distances, "r t -> r", "min").to("cpu")


if MAIN:
    num_pixels_y = 120
    num_pixels_z = 120
    y_limit = z_limit = 3
    rays = make_rays_2d_origin(num_pixels_y, num_pixels_z, y_limit, z_limit, t.tensor([-3.0, 0, 0]))
    intersections = raytrace_mesh_gpu(triangles, rays)
    picture = rearrange(intersections, "(y z) -> y z", y=num_pixels_y, z=num_pixels_z)

    fig = px.imshow(picture, origin="lower").update_layout(coloraxis_showscale=False)
    fig.show()
# %%
import math

def raytrace_mesh_lighting(
    triangles: t.Tensor, rays: t.Tensor, light: t.Tensor, ambient_intensity: float, device: str = "cpu"
) -> t.Tensor:
    '''For each ray, return the shade of the nearest triangle.
    triangles: shape (n_triangles, n_points=3, n_dims=3)
    rays: shape (n_pixels, n_points=2, n_dims=3)
    light: shape (n_dims=3, )
    device: The device to place tensors on.
    return: shape (n_pixels, )
    '''
    n_triangles = triangles.size(0)
    n_pixels = rays.size(0)
    triangles = triangles.to(device)
    rays = rays.to(device)
    light = light.to(device)

    matrices = t.zeros((n_pixels, n_triangles, 3, 3)).to(device)
    directions = rays[:, 1] - rays[:, 0]
    matrices[:, :, :, 0] = repeat(-directions, "r d -> r t d", t=n_triangles)
    matrices[:, :, :, 1] = repeat(triangles[:, 1] - triangles[:, 0], "t d -> r t d", r=n_pixels)
    matrices[:, :, :, 2] = repeat(triangles[:, 2] - triangles[:, 0], "t d -> r t d", r=n_pixels)
    bs = repeat(rays[:, 0], "r d -> r t d", t=n_triangles) - repeat(triangles[:, 0], "t d -> r t d", r=n_pixels)
    mask = t.linalg.det(matrices) != 0
    distances = t.full((n_pixels, n_triangles), float("inf")).to(device)
    solns = t.linalg.solve(matrices[mask], bs[mask])
    distances[mask] = t.where(
        (solns[:, 0] >= 0) & (solns[:, 1] >= 0) & (solns[:, 2] >= 0) & (solns[:, 1] + solns[:, 2] <= 1),
        solns[:, 0],
        t.tensor(float("inf")).to(device),
    )
    closest_triangle = distances.argmin(1)

    normals = t.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=1)
    normals = t.nn.functional.normalize(normals, p=2.0, dim=1)
    intensity = t.einsum("td,d->t", normals, light).gather(0, closest_triangle)
    side = t.einsum("rd,rd->r", normals.gather(0, repeat(closest_triangle, "r -> r d", d=3)), directions)
    intensity = t.maximum(t.sign(side) * intensity, t.zeros(())) + ambient_intensity
    intensity = t.where(
        distances.gather(1, closest_triangle.unsqueeze(1)).squeeze(1) == float("inf"),
        t.tensor(0.0).to(device),
        intensity,
    )

    return intensity.to("cpu")

def make_rays_camera(
    num_pixels_v: int,
    num_pixels_w: int,
    v_limit: float,
    w_limit: float,
    origin: t.Tensor,
    screen_distance: float,
    roll: float,
    pitch: float,
    yaw: float,
) -> t.Tensor:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.
    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''

    normal = t.tensor([math.cos(pitch) * math.cos(yaw), math.sin(pitch), math.cos(pitch) * math.sin(yaw)])
    w_vec = t.nn.functional.normalize(t.tensor([normal[2], 0, -normal[0]]), p=2.0, dim=0)
    v_vec = t.cross(normal, w_vec)
    w_vec_r = math.cos(roll) * w_vec + math.sin(roll) * v_vec
    v_vec_r = math.cos(roll) * v_vec - math.sin(roll) * w_vec

    rays = t.zeros((num_pixels_y, num_pixels_z, 2, 3))
    rays[:, :, 1, :] += repeat(origin + normal * screen_distance, "d -> w v d", w=num_pixels_w, v=num_pixels_v)
    rays[:, :, 1, :] += repeat(
        t.einsum("w, d -> w d", (t.arange(num_pixels_w) * 2.0 * w_limit / (num_pixels_w - 1) - w_limit), w_vec_r),
        "w d -> w v d",
        v=num_pixels_v,
    )
    rays[:, :, 1, :] += repeat(
        t.einsum("v, d -> v d", t.arange(num_pixels_v) * 2.0 * v_limit / (num_pixels_v - 1) - v_limit, v_vec_r),
        "v d -> w v d",
        w=num_pixels_w,
    )

    rays[:, :, 0, :] = origin
    return rearrange(rays, "y z n d -> (y z) n d", n=2, d=3)



if MAIN:
    num_pixels_y = 150
    num_pixels_z = 150
    y_limit = z_limit = 3
    rays = make_rays_2d_origin(num_pixels_y, num_pixels_z, y_limit, z_limit, t.tensor([-3.0, 0, 0]))
    light = t.tensor([0.0, -1.0, 1.0])
    ambient_intensity = 0.5
    intersections = raytrace_mesh_lighting(triangles, rays, light, ambient_intensity, "cuda")
    picture = rearrange(intersections, "(y z) -> y z", y=num_pixels_y, z=num_pixels_z)

    fig = px.imshow(picture, origin="lower", labels={"x": "X", "y": "Y"}, color_continuous_scale="magma").update_layout(coloraxis_showscale=False)
    fig.show()
# %%
def get_random_rotation_matrix(N, theta_max=t.pi):
    mat = t.eye(N)
    for i in range(N):
        rot_mat = t.eye(N)
        theta = (t.rand(1) - 0.5) * theta_max
        rot_mat_2d = t.tensor([
            [t.cos(theta), -t.sin(theta)], 
            [t.sin(theta), t.cos(theta)]
        ])
        if i == N - 1:
            rot_mat[[-1, -1, 0, 0], [-1, 0, -1, 0]] = rot_mat_2d.flatten()
        else:
            rot_mat[i :i+2, i :i+2] = rot_mat_2d
        mat = mat @ rot_mat
    return mat

if MAIN:
    num_pixels_y = 150
    num_pixels_z = 150
    y_limit = z_limit = 3

    rays = make_rays_camera(num_pixels_y, num_pixels_z, y_limit, z_limit, t.tensor([-1.0, 3.0, 0.0]), 3.0, 0.0, -1.0, 0.0)
    light = t.tensor([0.0, -1.0, 1.0])
    ambient_intensity = 0.5
    intersections = raytrace_mesh_lighting(triangles, rays, light, ambient_intensity, "cuda")
    picture = rearrange(intersections, "(y z) -> y z", y=num_pixels_y, z=num_pixels_z)
    fig = px.imshow(picture, origin="lower", labels={"x": "X", "y": "Y"}, color_continuous_scale="magma").update_layout(coloraxis_showscale=False)
    fig.show()
# %%
def raytrace_mesh_lambert(triangles: t.Tensor, rays: t.Tensor) -> t.Tensor:
    '''For each ray, return the distance to the closest intersecting triangle, or infinity.
    triangles: shape (n_triangles, n_points=3, n_dims=3)
    rays: shape (n_pixels, n_points=2, n_dims=3)
    return: shape (n_pixels, )
    '''
    # triangles = [triangle, point, coord]
    # rays = [pixel, orig_dir, coord]

    n_triangles = len(triangles)
    n_pixels = len(rays)

    rep_triangles = einops.repeat(triangles, "triangle point coord -> pixel triangle point coord", pixel=n_pixels)
    rep_rays = einops.repeat(rays, "pixel orig_dir coord -> pixel triangle orig_dir coord", triangle=n_triangles)

    O = rep_rays[:, :, 0, :]  # [pixel, triangle, coord]
    D = rep_rays[:, :, 1, :]  # [pixel, triangle, coord]
    A = rep_triangles[:, :, 0, :]  # [pixel, triangle, coord]
    B = rep_triangles[:, :, 1, :]  # [pixel, triangle, coord]
    C = rep_triangles[:, :, 2, :]  # [pixel, triangle, coord]
    rhs = O - A  # [pixel, triangle, coord]
    lhs = t.stack([-D, B - A, C - A], dim=3)  # [pixel, triangle, coord, suv]
    dets = t.linalg.det(lhs)  # [pixel, triangle]
    dets = dets < 1e-5
    eyes = t.einsum("i j , k l -> i j k l", [dets, t.eye(3)])
    lhs += eyes
    results = t.linalg.solve(lhs, rhs)  # [pixel, triangle, suv]
    intersects = (
        ((results[:, :, 1] + results[:, :, 2]) <= 1)
        & (results[:, :, 0] >= 0)
        & (results[:, :, 1] >= 0)
        & (results[:, :, 2] >= 0)
        & (dets == False)
    )  # [pixel, triangle]
    distances = t.where(intersects, results[:, :, 0].double(), t.inf)  # [pixel, triangle]

    # Lambert shading (dot product of triangle's normal vector with light direction)
    indices = t.argmin(distances, dim=1)
    tri_vecs1 = triangles[:, 0, :] - triangles[:, 1, :]
    tri_vecs2 = triangles[:, 1, :] - triangles[:, 2, :]
    normvecs = t.cross(tri_vecs1, tri_vecs2, dim=1)  # [triangle coord]
    normvecs -= normvecs.min(1, keepdim=True)[0]
    normvecs /= normvecs.max(1, keepdim=True)[0]
    lightvec = t.tensor([[0.0, 1.0, 1.0]] * n_triangles)
    tri_lights = abs(t.einsum("t c , t c -> t", [normvecs, lightvec]))  # triangle
    pixel_lights = 1.0 / (einops.reduce(distances, "pixel triangle -> pixel", "min")) ** 2
    pixel_lights *= tri_lights[indices]
    return pixel_lights



if MAIN:
    rot_mat = get_random_rotation_matrix(N=3, theta_max=t.pi/3)
    num_pixels_y = 200
    num_pixels_z = 200
    y_limit = z_limit = 1
    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0, 0] = -2
    rays[0, :, 0]
    result = raytrace_mesh_lambert(t.einsum("i j k, k l -> i j l", [triangles, rot_mat]), rays)
    result = result.reshape(num_pixels_y, num_pixels_z)
    fig = px.imshow(result, origin="lower", labels={"x": "X", "y": "Y"}, color_continuous_scale="magma").update_layout(coloraxis_showscale=False)
    fig.show()
# %%
def raytrace_mesh_lambert_wireframe(triangles: t.Tensor, rays: t.Tensor, triangle_perim: float = 0) -> t.Tensor:
    '''For each ray, return the distance to the closest intersecting triangle, or infinity.
    triangles: shape (n_triangles, n_points=3, n_dims=3)
    rays: shape (n_pixels, n_points=2, n_dims=3)
    return: shape (n_pixels, )
    '''
    # triangles = [triangle, point, coord]
    # rays = [pixel, orig_dir, coord]

    n_triangles = len(triangles)
    n_pixels = len(rays)

    rep_triangles = einops.repeat(triangles, "triangle point coord -> pixel triangle point coord", pixel=n_pixels)
    rep_rays = einops.repeat(rays, "pixel orig_dir coord -> pixel triangle orig_dir coord", triangle=n_triangles)

    O = rep_rays[:, :, 0, :]  # [pixel, triangle, coord]
    D = rep_rays[:, :, 1, :]  # [pixel, triangle, coord]
    A = rep_triangles[:, :, 0, :]  # [pixel, triangle, coord]
    B = rep_triangles[:, :, 1, :]  # [pixel, triangle, coord]
    C = rep_triangles[:, :, 2, :]  # [pixel, triangle, coord]
    rhs = O - A  # [pixel, triangle, coord]
    lhs = t.stack([-D, B - A, C - A], dim=3)  # [pixel, triangle, coord, suv]
    dets = t.linalg.det(lhs)  # [pixel, triangle]
    dets = dets < 1e-5
    eyes = t.einsum("i j , k l -> i j k l", [dets, t.eye(3)])
    lhs += eyes
    results = t.linalg.solve(lhs, rhs)  # [pixel, triangle, suv]
    intersects = (
        ((results[:, :, 1] + results[:, :, 2]) <= 1)
        & (results[:, :, 0] >= 0.0)
        & (results[:, :, 1] >= 0.0)
        & (results[:, :, 2] >= 0.0)
        & (dets == False)
    )  # [pixel, triangle]
    intersects_perim = (
        ((results[:, :, 1] + results[:, :, 2]) >= 1 - triangle_perim)
        | (results[:, :, 1] <= triangle_perim)
        | (results[:, :, 2] <= triangle_perim)
    )
    intersects = intersects & intersects_perim
    distances = t.where(intersects, results[:, :, 0].double(), t.inf)  # [pixel, triangle]

    # Lambert shading (dot product of triangle's normal vector with light direction)
    indices = t.argmin(distances, dim=1)
    tri_vecs1 = triangles[:, 0, :] - triangles[:, 1, :]
    tri_vecs2 = triangles[:, 1, :] - triangles[:, 2, :]
    normvecs = t.cross(tri_vecs1, tri_vecs2, dim=1)  # [triangle coord]
    normvecs -= normvecs.min(1, keepdim=True)[0]
    normvecs /= normvecs.max(1, keepdim=True)[0]
    lightvec = t.tensor([[0.0, 1.0, 1.0]] * n_triangles)
    tri_lights = abs(t.einsum("t c , t c -> t", [normvecs, lightvec]))  # triangle
    pixel_lights = 1.0 / (einops.reduce(distances, "pixel triangle -> pixel", "min")) ** 2
    pixel_lights *= tri_lights[indices]
    return pixel_lights


if MAIN:
    rot_mat = get_random_rotation_matrix(N=3, theta_max=t.pi/4)
    num_pixels_y = 200
    num_pixels_z = 200
    y_limit = z_limit = 1

    triangle_perim = 0.1

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0, 0] = -2
    rays[0, :, 0]
    result = raytrace_mesh_lambert_wireframe(t.einsum("i j k, k l -> i j l", [triangles, rot_mat]), rays, triangle_perim)
    result = result.reshape(num_pixels_y, num_pixels_z)
    fig = px.imshow(result, origin="lower", labels={"x": "X", "y": "Y"}, color_continuous_scale="magma").update_layout(coloraxis_showscale=False)
    fig.show()
# %%
