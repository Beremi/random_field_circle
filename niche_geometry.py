"""
Utilities for handling the niche boundary and morphing the circular
tunnel random field to the measured polygonal shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import meshio
import numpy as np


Point = Tuple[float, float]
Segment = Tuple[Point, Point]


def load_niche_segments(filename: str) -> List[Segment]:
    """
    Read a VTK/VTU file and extract all 2D line segments describing
    the niche boundary. Quadratic line elements (line3) are split into
    two linear segments to retain curvature.
    """
    mesh = meshio.read(filename)
    points = mesh.points[:, :2]
    segments: List[Segment] = []

    for cell_block in mesh.cells:
        ctype = cell_block.type
        cells = cell_block.data

        if ctype in ("line", "line2"):
            for i0, i1 in cells:
                p0 = tuple(points[i0])
                p1 = tuple(points[i1])
                segments.append((p0, p1))

        elif ctype == "line3":
            for i0, i1, i2 in cells:
                p0, pm, p1 = points[[i0, i1, i2]]
                segments.append((tuple(p0), tuple(pm)))
                segments.append((tuple(pm), tuple(p1)))

    return segments


def segments_to_unique_points(segments: Sequence[Segment]) -> np.ndarray:
    """
    Convert a collection of segments to unique 2D points.
    """
    unique = {tuple(map(float, p)) for seg in segments for p in seg}
    points = np.array(list(unique))
    return points


def sort_points_by_angle(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort 2D points by their polar angle around the origin. Assumes the
    niche is roughly star-shaped with respect to (0,0).
    """
    theta = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    order = np.argsort(theta)
    return theta[order], points[order]


def _thin_plate_phi(r: np.ndarray) -> np.ndarray:
    """
    Thin-plate spline radial basis φ(r) = r² log r with φ(0) = 0.
    """
    safe_r = np.maximum(r, 1e-12)
    phi = safe_r**2 * np.log(safe_r)
    phi[r == 0.0] = 0.0
    return phi


@dataclass
class CircleToPolygonMorpher:
    """
    Mesh morphing map that warps the circular tunnel boundary to the
    polygon described by the provided segments using a thin-plate RBF.
    """

    boundary_points: np.ndarray
    circle_radius: float = 1.0
    n_boundary_ctrl: int = 180
    n_outer_ctrl: int = 40
    outer_radius: float | None = None
    regularization: float = 1e-6

    def __post_init__(self) -> None:
        theta, points = sort_points_by_angle(self.boundary_points)
        self._theta_sorted = theta
        self._points_sorted = points
        self._theta_start = float(theta[0])
        self._theta_ext = np.concatenate([theta, [theta[0] + 2.0 * np.pi]])
        self._x_ext = np.concatenate([points[:, 0], [points[0, 0]]])
        self._y_ext = np.concatenate([points[:, 1], [points[0, 1]]])

        if self.outer_radius is None:
            r_max = np.max(np.linalg.norm(points, axis=1))
            self.outer_radius = 1.4 * r_max

        self._centroid = points.mean(axis=0)
        self._build_rbf()

    def sample_boundary(self, theta: np.ndarray) -> np.ndarray:
        """
        Evaluate the polygon boundary at given polar angles using a
        periodic linear interpolant.
        """
        theta = np.mod(theta, 2.0 * np.pi)
        mask = theta < self._theta_start
        if np.any(mask):
            theta = theta.copy()
            theta[mask] += 2.0 * np.pi

        x = np.interp(theta, self._theta_ext, self._x_ext)
        y = np.interp(theta, self._theta_ext, self._y_ext)
        return np.column_stack([x, y])

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Apply the RBF morph to an array of 2D points.
        """
        points_arr = np.asarray(points, dtype=float)
        pts = np.atleast_2d(points_arr)
        diff = pts[:, None, :] - self._src_ctrl[None, :, :]
        r = np.linalg.norm(diff, axis=2)
        Phi = _thin_plate_phi(r)
        disp = Phi @ self._weights
        warped = pts + disp
        return warped.reshape(points_arr.shape)

    def warp_grid(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp a rectangular grid defined by (x, y) vectors to the polygon.
        Returns X_warped, Y_warped arrays matching the shape of log k.
        """
        X, Y = np.meshgrid(x, y, indexing="xy")
        pts = np.column_stack([X.ravel(), Y.ravel()])
        warped = self.transform_points(pts)
        Xw = warped[:, 0].reshape(X.shape)
        Yw = warped[:, 1].reshape(Y.shape)
        return Xw, Yw

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_rbf(self) -> None:
        """
        Assemble control points and fit the thin-plate RBF weights.
        """
        theta_ctrl = np.linspace(0.0, 2.0 * np.pi, self.n_boundary_ctrl, endpoint=False)
        circle_pts = np.column_stack(
            [
                self.circle_radius * np.cos(theta_ctrl),
                self.circle_radius * np.sin(theta_ctrl),
            ]
        )
        polygon_pts = self.sample_boundary(theta_ctrl)

        controls_src = [circle_pts]
        controls_tgt = [polygon_pts]

        # Anchor the tunnel center to the polygon centroid
        controls_src.append(np.array([[0.0, 0.0]]))
        controls_tgt.append(self._centroid[None, :])

        # Keep the far field nearly fixed
        if self.n_outer_ctrl > 0:
            theta_outer = np.linspace(
                0.0, 2.0 * np.pi, self.n_outer_ctrl, endpoint=False
            )
            outer = np.column_stack(
                [
                    self.outer_radius * np.cos(theta_outer),
                    self.outer_radius * np.sin(theta_outer),
                ]
            )
            controls_src.append(outer)
            controls_tgt.append(outer)

        self._src_ctrl = np.vstack(controls_src)
        self._tgt_ctrl = np.vstack(controls_tgt)
        self._weights = self._fit_weights()

    def _fit_weights(self) -> np.ndarray:
        diff = self._src_ctrl[:, None, :] - self._src_ctrl[None, :, :]
        r = np.linalg.norm(diff, axis=2)
        K = _thin_plate_phi(r)
        K += self.regularization * np.eye(K.shape[0])

        disp = self._tgt_ctrl - self._src_ctrl
        wx = np.linalg.solve(K, disp[:, 0])
        wy = np.linalg.solve(K, disp[:, 1])
        return np.column_stack([wx, wy])


def plot_segments(ax, segments: Sequence[Segment], **plot_kwargs):
    """
    Convenience helper to draw the polygon boundary onto an axis.
    """
    for (x0, y0), (x1, y1) in segments:
        ax.plot([x0, x1], [y0, y1], **plot_kwargs)
    return ax
