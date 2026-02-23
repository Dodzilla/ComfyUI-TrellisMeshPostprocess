"""ComfyUI custom nodes for Trellis mesh postprocessing."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if not np.isfinite(length) or length <= 1e-12:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return v / length


def _position_key(vertices: np.ndarray, index: int, epsilon: float) -> str:
    p = vertices[index]
    qx = int(round(float(p[0]) / epsilon))
    qy = int(round(float(p[1]) / epsilon))
    qz = int(round(float(p[2]) / epsilon))
    return f"{qx},{qy},{qz}"


@dataclass
class _Cluster:
    avg: np.ndarray
    count: int


class PixelArtistryTrellisMeshPostprocessNormals:
    """
    Recalculate normals for duplicated/split Trellis vertices by clustering
    face normals in quantized position space.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "position_epsilon": (
                    "FLOAT",
                    {
                        "default": 0.00001,
                        "min": 0.00000001,
                        "max": 0.01,
                        "step": 0.000001,
                    },
                ),
                "normal_crease_deg": (
                    "FLOAT",
                    {
                        "default": 55.0,
                        "min": 1.0,
                        "max": 180.0,
                        "step": 0.5,
                    },
                ),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2/Mesh"

    def _assign_face_cluster(
        self,
        groups: Dict[str, List[_Cluster]],
        pos_key: str,
        face_normal: np.ndarray,
        min_dot: float,
    ) -> int:
        cluster_list = groups.get(pos_key)
        if cluster_list is None:
            cluster_list = []
            groups[pos_key] = cluster_list

        for idx, cluster in enumerate(cluster_list):
            if cluster.count <= 0:
                continue

            dot = float(np.dot(cluster.avg, face_normal))
            if dot >= min_dot:
                next_count = cluster.count + 1
                cluster.avg = _normalize((cluster.avg * cluster.count + face_normal) / next_count)
                cluster.count = next_count
                return idx

        cluster_list.append(_Cluster(avg=face_normal.copy(), count=1))
        return len(cluster_list) - 1

    def process(self, trimesh, position_epsilon: float, normal_crease_deg: float):
        mesh = trimesh.copy()
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            return (mesh,)

        if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
            return (mesh,)

        epsilon = max(1e-8, min(1e-2, float(position_epsilon)))
        crease = max(1.0, min(180.0, float(normal_crease_deg)))
        min_dot = float(np.cos(np.deg2rad(crease)))

        vertex_count = int(vertices.shape[0])
        position_keys = [_position_key(vertices, i, epsilon) for i in range(vertex_count)]
        groups_by_position: Dict[str, List[_Cluster]] = {}
        preferred_cluster_by_vertex: List[Dict[int, float]] = [defaultdict(float) for _ in range(vertex_count)]
        accum: Dict[str, np.ndarray] = {}

        for face in faces:
            i0, i1, i2 = (int(face[0]), int(face[1]), int(face[2]))
            if i0 < 0 or i1 < 0 or i2 < 0:
                continue
            if i0 >= vertex_count or i1 >= vertex_count or i2 >= vertex_count:
                continue

            p0 = vertices[i0]
            p1 = vertices[i1]
            p2 = vertices[i2]

            face_vec = np.cross(p1 - p0, p2 - p0)
            magnitude = float(np.linalg.norm(face_vec))
            if not np.isfinite(magnitude) or magnitude <= 1e-12:
                continue

            face_normal = face_vec / magnitude

            for vidx in (i0, i1, i2):
                pos_key = position_keys[vidx]
                cluster_index = self._assign_face_cluster(groups_by_position, pos_key, face_normal, min_dot)
                key = f"{pos_key}|{cluster_index}"

                if key in accum:
                    accum[key] += face_vec
                else:
                    accum[key] = face_vec.copy()

                preferred_cluster_by_vertex[vidx][cluster_index] += magnitude

        fallback_normals = None
        try:
            current_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
            if current_normals.shape == vertices.shape:
                fallback_normals = current_normals
        except Exception:
            fallback_normals = None

        normals = np.zeros_like(vertices, dtype=np.float64)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        for vidx in range(vertex_count):
            pos_key = position_keys[vidx]
            preferred = preferred_cluster_by_vertex[vidx]
            cluster_index = 0

            if preferred:
                cluster_index = max(preferred.items(), key=lambda item: item[1])[0]

            key = f"{pos_key}|{cluster_index}"
            summed = accum.get(key)

            if summed is None:
                if fallback_normals is not None:
                    summed = fallback_normals[vidx]
                else:
                    summed = up

            normals[vidx] = _normalize(summed)

        mesh.vertex_normals = normals.astype(np.float32)
        return (mesh,)


NODE_CLASS_MAPPINGS = {
    "PixelArtistryTrellisMeshPostprocessNormals": PixelArtistryTrellisMeshPostprocessNormals,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelArtistryTrellisMeshPostprocessNormals": "Trellis2 - Mesh Postprocess Normals",
}
