"""Optional Warp mesh queries; imported only after selecting the Warp backend."""

import numpy as np
import warp as wp


@wp.kernel
def _mesh_signed_distances(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    maximum_distance: wp.float32,
    signed_distances: wp.array(dtype=wp.float32),
):
    query_index = wp.tid()
    query_point = query_points[query_index]
    result = wp.mesh_query_point_sign_parity(mesh_id, query_point, maximum_distance, 3, 0.01)
    if result.result:
        closest_point = wp.mesh_eval_position(mesh_id, result.face, result.u, result.v)
        signed_distances[query_index] = wp.length(query_point - closest_point) * result.sign
    else:
        signed_distances[query_index] = 1.0e30


class WarpDistanceQuery:
    """Own a normalized mesh BVH until explicit, synchronized release.

    ``prepare_distance_grid`` serializes the complete lifecycle of this internal
    query. Direct callers must also manage concurrency and use close or a context
    manager; this object does not promise thread-safe shared access.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, device: str):
        if device not in {"cpu", "cuda"}:
            raise ValueError("Warp query device must be cpu or cuda")
        self.device = device
        self.vertex_buffer = None
        self.index_buffer = None
        self.mesh = None
        self._closed = False
        try:
            wp.init()
            if device == "cuda" and not wp.is_cuda_available():
                raise ValueError("CUDA Warp queries requested, but Warp has no CUDA device")
            self.mesh_radius_bound = float(np.linalg.norm(vertices, axis=1).max())
            self.vertex_buffer = wp.array(vertices, dtype=wp.vec3, device=device)
            self.index_buffer = wp.array(
                np.asarray(faces, dtype=np.int32).ravel(), dtype=wp.int32, device=device
            )
            self.mesh = wp.Mesh(points=self.vertex_buffer, indices=self.index_buffer)
        except BaseException as initialization_error:
            try:
                self.close()
            except Exception as cleanup_error:
                raise initialization_error from cleanup_error
            raise

    def signed_distances(self, normalized_points: np.ndarray) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Warp distance query is closed")
        maximum_distance = (
            float(np.linalg.norm(normalized_points, axis=1).max()) + self.mesh_radius_bound + 1
        )
        query_buffer = None
        distances = None
        completed = False
        try:
            query_buffer = wp.array(normalized_points, dtype=wp.vec3, device=self.device)
            distances = wp.empty(len(normalized_points), dtype=wp.float32, device=self.device)
            wp.launch(
                _mesh_signed_distances,
                dim=len(normalized_points),
                inputs=[self.mesh.id, query_buffer, maximum_distance],
                outputs=[distances],
                device=self.device,
            )
            # Warp CPU .numpy() aliases a Warp allocation. Copy before returning
            # so no such allocation can outlive the preparation's lifecycle lock.
            result = distances.numpy().copy()
            completed = True
            return result
        finally:
            try:
                if not completed and (query_buffer is not None or distances is not None):
                    wp.synchronize_device(self.device)
            finally:
                distances = None
                query_buffer = None

    def close(self) -> None:
        """Finish outstanding work before releasing the BVH and backing arrays."""
        if self._closed:
            return
        self._closed = True
        try:
            if (
                self.vertex_buffer is not None
                or self.index_buffer is not None
                or self.mesh is not None
            ):
                wp.synchronize_device(self.device)
        finally:
            self.mesh = None
            self.index_buffer = None
            self.vertex_buffer = None

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Warp distance query is closed")
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()
