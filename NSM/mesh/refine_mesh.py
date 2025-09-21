import numpy as np
import pyvista as pv
from .triangle_metrics import TriangleProperties
from vtk.util.numpy_support import numpy_to_vtk


def midpoint(vertex1, vertex2):
    """
    Get the midpoint between two vertices.

    Parameters:
    - vertex1: A numpy array of the xyz position of the first vertex.
    - vertex2: A numpy array of the xyz position of the second vertex.

    Returns:
    - midpoint: A numpy array of the xyz position of the midpoint between the two vertices.
    """
    return (vertex1 + vertex2) / 2


def get_faces(mesh):
    """
    Get the faces of a mesh.

    Parameters:
    - mesh: A PyVista mesh.

    Returns:
    - faces: A numpy array of vertex indices for each face (Nx3).
    """
    return mesh.faces.reshape(-1, 4)[:, 1:]


def find_all_faces_to_split(mesh, cells_to_divide):
    """
    Find all faces to fully split (into 4 sub triangles). This is done by
    adding any faces that are adjacent to > 1 face in the cells_to_divide.
    If these faces are not added to cells_to_divide, then there will be
    an error during face splitting.

    Parameters:
    - mesh: A PyVista mesh.
    - cells_to_divide: A numpy array of indices of faces to split.

    Returns:
    - cells_to_divide: A numpy array of indices of faces to split.
    - list_adjacent: A list of indices of faces that are adjacent to the faces in cells_to_divide.
    """

    list_adjacent = []

    faces = get_faces(mesh)

    unique = 0
    not_unique = 0

    cells_to_divide = cells_to_divide.tolist()

    for face_idx in cells_to_divide:
        for edge_idx in range(3):
            p1 = faces[face_idx, edge_idx]
            p2 = faces[face_idx, (edge_idx + 1) % 3]
            adjacent_face_idx = find_adjacent_face(faces=faces, p1=p1, p2=p2, current_face=face_idx)
            if adjacent_face_idx not in list_adjacent:
                list_adjacent.append(adjacent_face_idx)
                unique += 1
            elif adjacent_face_idx not in cells_to_divide:
                cells_to_divide.append(adjacent_face_idx)
                not_unique += 1

    cells_to_divide = np.array(cells_to_divide)

    return cells_to_divide


def new_vertices_faces(mesh, face_idx, new_vertices, cells_to_divide):
    """
    Generate new vertices and faces for a single face that is being split.

    Parameters:
    - mesh: A PyVista mesh.
    - face_idx: The index of the face to split.
    - new_vertices: A list to store new vertices added to the mesh.
    - cells_to_divide: A numpy array of indices of faces to split.

    Returns:
    - new_vertices: A list of new vertices added to the mesh.
    - new_faces: A list of new faces (triangles), each represented as a list of vertex indices.
    - adjacent_face_indices: A list of indices of new faces that are adjacent to the face being split (to be used later to delete the original face and replace it with the new faces)
    """
    # get the faces & points/vertices from the mesh.
    faces = get_faces(mesh)
    points = mesh.points

    # preallocate lists to store the new vertex indices indices.
    midpoint_indices = []

    # store dict of info for each edge of the face (point indices, xyz position, midpoint position)
    edges = {}
    for i in range(3):
        p1 = faces[face_idx, i]
        v1 = points[p1]
        p2 = faces[face_idx, (i + 1) % 3]
        v2 = points[p2]
        mid = midpoint(v1, v2)
        edges[i] = {"p1": p1, "p2": p2, "v1": v1, "v2": v2, "midpoint": mid}

    # preallocate lists to store the new faces and adjacent face indices
    new_adjacent_faces = []
    adjacent_face_indices = []

    for edge_idx, edge_dict in edges.items():
        # for each midpoint_ (xyz position/vertex) check to see if it exists
        # if it does, return that index in the current mesh.points & leave the
        # new_vertices alone. If it doesnt, then add the vertex to new_vertices
        # & return the index (based on the total number of points that already
        # exist)
        new_vertices, midpoint_index = add_vertex_if_new(edge_dict["midpoint"], mesh, new_vertices)
        # store the index so it can be used to create the new faces
        midpoint_indices.append(midpoint_index)

        # get adjacent faces to the one being split (shares the edge of this loop)
        adjacent_face_idx = find_adjacent_face(
            faces=faces, p1=edge_dict["p1"], p2=edge_dict["p2"], current_face=face_idx
        )
        # store the index so it can be deleted later (we want to replace existing one with the 2 new ones)
        if adjacent_face_idx not in cells_to_divide:
            # if the adjacent face is not in the list of cells to divide, then split this face
            # and add the new faces to the list of new faces.
            adjacent_face_indices.append(adjacent_face_idx)
            # create the 2 new faces and add them to the list of new faces.
            new_adjacent_faces.extend(
                create_new_adjacent_faces(
                    original_vertex_indices=faces[adjacent_face_idx],
                    p1=edge_dict["p1"],
                    p2=edge_dict["p2"],
                    midpoint_idx=midpoint_index,
                )
            )

    # Create new faces using original vertices and new midpoints
    new_faces = create_new_faces(faces[face_idx], midpoint_indices)  # Implement this
    # combine the two lists of new faces
    new_faces.extend(new_adjacent_faces)

    return new_vertices, new_faces, adjacent_face_indices


def create_new_adjacent_faces(original_vertex_indices, p1, p2, midpoint_idx):
    """
    Split a triangle into two by drawing a line from the midpoint of one edge to the opposite vertex, preserving normals.

    Parameters:
    - original_vertex_indices: tuple/list of vertex indices for the original triangle
    - p1: one vertex index of the edge to be split
    - p2: the other vertex index of the edge to be split
    - midpoint_idx: index of the midpoint on the edge between p1 and p2

    Returns:
    - new_faces: A list of two new faces (triangles), each represented as a list of vertex indices,
                 ensuring the correct winding order is preserved.
    """

    # Find the third vertex that is not part of the edge to be split

    original_vertex_indices = np.squeeze(original_vertex_indices)

    third_vertex = [idx for idx in original_vertex_indices if idx not in (p1, p2)][0]

    # Assuming original_vertex_indices were provided in a CCW order,
    # and without loss of generality, assuming (p1, midpoint_idx, p2) maintains the CCW orientation
    # We need to ensure new triangles are created with the right order
    new_faces = [
        [p1, midpoint_idx, third_vertex],  # Triangle 1: maintaining CCW order
        [midpoint_idx, p2, third_vertex],  # Triangle 2: adjusting order to maintain CCW
    ]

    return new_faces


def find_faces_with_edge(faces, p1, p2):
    """
    Find the indices of faces that contain a given edge.

    Parameters:
    - faces: A numpy array of vertex indices for each face (Nx3).
    - p1: The index of the first vertex of the edge.
    - p2: The index of the second vertex of the edge.

    Returns:
    - face_indices: A numpy array of indices of faces that contain the edge.
    """

    p1_faces = np.sum(faces == p1, axis=1)
    p2_faces = np.sum(faces == p2, axis=1)

    face_indices = np.where(p1_faces * p2_faces)[0]

    return face_indices


def find_adjacent_face(faces, p1, p2, current_face):
    """
    Find the index of the face that is adjacent to a given face, sharing a given edge.

    Parameters:
    - faces: A numpy array of vertex indices for each face (Nx3).
    - p1: The index of the first vertex of the edge.
    - p2: The index of the second vertex of the edge.
    - current_face: The index of the current face.

    Returns:
    - face_index: The index of the adjacent face.
    """
    face_indices = find_faces_with_edge(faces, p1, p2)

    assert (
        len(face_indices) == 2
    ), f"Number of faces should be 2, found {len(face_indices)}: {face_indices}"

    face_index = face_indices[np.where(face_indices != current_face)]

    assert len(face_index) == 1

    return face_index[0]


def add_vertex_if_new(
    vertex, mesh, new_vertices, threshold=1e-10
):  # np.finfo(float).eps - this is ~ 2.3 e-16
    """
    Add a vertex to the new_vertices list if it doesn't already exist in the mesh or new_vertices.

    Parameters:
    - vertex: The new vertex to add (numpy array or list of 3 floats: x, y, z).
    - mesh: The original PyVista mesh.
    - new_vertices: A list to store new vertices added to the mesh.

    Returns:
    - The index of the vertex in the combined list of original mesh vertices and new_vertices.
    """

    # Combine original mesh vertices with new_vertices to search for existing vertex
    if len(new_vertices) > 0:
        combined_vertices = np.vstack((mesh.points, np.array(new_vertices)))
    else:
        combined_vertices = mesh.points

    # Check if the vertex already exists in the combined vertices array
    distances = np.linalg.norm(combined_vertices - vertex, axis=1)
    existing_vertex_index = np.where(distances < threshold)[
        0
    ]  # Use a small threshold to determine equality

    if len(existing_vertex_index) > 0:
        # Vertex exists, return the index of the existing vertex
        return new_vertices, existing_vertex_index[0]

    # Vertex is new, add to new_vertices and return its new index
    new_vertices.append(vertex)
    return new_vertices, len(combined_vertices)


def create_new_faces(original_vertex_indices, midpoint_indices):
    """
    Create new faces (triangles) by subdividing an original triangle using midpoints on its edges.

    Parameters:
    - original_vertex_indices: Indices of the vertices of the original triangle.
    - midpoint_indices: Indices of the midpoints on the edges of the original triangle.

    Returns:
    - new_faces: A list of new faces (triangles), where each face is represented as a list of vertex indices.
    """
    # Original vertices are A, B, C
    A, B, C = original_vertex_indices
    # Midpoints are AB, BC, CA (corresponding to edges AB, BC, and CA)
    AB, BC, CA = midpoint_indices

    # Create new faces using the original vertices and the new midpoints
    # The new triangles will be (A, AB, CA), (AB, B, BC), (CA, BC, C), and (AB, BC, CA)
    new_faces = [[A, AB, CA], [AB, B, BC], [CA, BC, C], [AB, BC, CA]]

    return new_faces


def update_mesh(mesh, new_vertices, new_faces, faces_to_delete):
    """
    Create new mesh by combining new_vertices/new_faces with the existing mesh
    faces/vertices and delete the faces for all of the cells_to_divide
    (because these have now been subdivided).

    Parameters:
    - mesh: A PyVista mesh.
    - new_vertices: A list of new vertices added to the mesh.
    - new_faces: A list of new faces (triangles), each represented as a list of vertex indices.
    - faces_to_delete: A numpy array of indices of faces to delete.

    Returns:
    - mesh: A new PyVista mesh with the new vertices and faces.
    """
    # Update the mesh with new vertices and faces

    faces_orig = get_faces(mesh)

    if len(new_faces) > 0:
        faces = np.concatenate((faces_orig, new_faces), axis=0)
        faces = np.delete(faces, faces_to_delete, axis=0)
        faces = np.hstack((np.ones((faces.shape[0], 1), dtype=int) * 3, faces))

        if len(new_vertices) > 0:
            new_pts = np.concatenate((mesh.points, new_vertices), axis=0)
        else:
            new_pts = mesh.points

        mesh = pv.PolyData(new_pts, faces)
    else:
        mesh = pv.PolyData(mesh, deep=True)

    return mesh


def subdivide_triangles(mesh, cells_to_divide):
    """
    Subdivide triangles in a mesh by splitting specified cells (faces/triangles)
    into 4 sub-triangles.

    Parameters:
    - mesh: A PyVista mesh.
    - cells_to_divide: A numpy array of indices of faces to split.

    Returns:
    - mesh: A new PyVista mesh with the specified cells split into 4 sub-triangles.
    """
    # Placeholder for new vertices and faces
    new_vertices = []
    new_faces = []
    faces_to_delete = []

    cells_to_divide = find_all_faces_to_split(mesh, cells_to_divide)

    for idx, cell_idx in enumerate(cells_to_divide):
        new_vertices, new_faces_, faces_to_delete_ = new_vertices_faces(
            mesh, cell_idx, new_vertices, cells_to_divide
        )
        new_faces.extend(new_faces_)
        faces_to_delete.extend(faces_to_delete_)

    faces_to_delete.extend(cells_to_divide)

    # create a new mesh by combining new_vertices/new_faces
    # with the existing mesh faces/vertices and delete the
    # faces for all of the cells_to_divide (because these
    # have now been subdivided).
    mesh_ = update_mesh(
        mesh, new_vertices, new_faces, faces_to_delete
    )  # Implement mesh update logic

    return mesh_


def get_target_cells(
    mesh, area_threshold=None, length_threshold=None, max_length_threshold=None, verbose=False
):
    """
    Get the indices of cells (faces/triangles) in a mesh that meet the criteria for subdivision.

    Parameters:
    - mesh: A PyVista mesh.
    - area_threshold: The maximum area of a triangle before it is subdivided.
    - length_threshold: The maximum:min edge length ratio of a triangle before it is subdivided.
    - max_length_threshold: The maximum edge length of a triangle before it is subdivided.

    Returns:
    - cells_to_divide: A numpy array of indices of faces to split.

    """
    triangle_properties = TriangleProperties(mesh)

    areas = triangle_properties.areas(norm=True)

    if area_threshold is not None:
        areas_binary = areas > area_threshold
    else:
        areas_binary = np.zeros_like(areas)

    edge_ratio = triangle_properties.edge_ratio()

    if length_threshold is not None:
        edge_ratio_binary = edge_ratio > length_threshold
    else:
        edge_ratio_binary = np.zeros_like(edge_ratio)

    max_lengths = triangle_properties.edge_length_max()

    if max_length_threshold is not None:
        max_length_binary = max_lengths > max_length_threshold
    else:
        max_length_binary = np.zeros_like(max_length_binary)

    cells_to_divide_binary = np.max((edge_ratio_binary, areas_binary, max_length_binary), axis=0)
    cells_to_divide = np.where(cells_to_divide_binary)[0]

    if verbose is True:
        print(sum(edge_ratio_binary), edge_ratio_binary.shape)
        print(sum(areas_binary), areas_binary.shape)
        print(cells_to_divide.shape)

    return cells_to_divide


def subdivide_large_triangles(
    mesh, area_threshold=None, length_threshold=None, max_length_threshold=None, verbose=False
):
    """
    Subdivide large triangles in a mesh by splitting specified cells (faces/triangles)

    Parameters:
    - mesh: A PyVista mesh.
    - area_threshold: The maximum area of a triangle before it is subdivided.
    - length_threshold: The maximum:min edge length ratio of a triangle before it is subdivided.
    - max_length_threshold: The maximum edge length of a triangle before it is subdivided.
    - verbose: Whether to print additional information.

    Returns:
    - mesh_: A new PyVista mesh with the specified cells split into 4 sub-triangles.
    """

    cells_to_divide = get_target_cells(
        mesh, area_threshold, length_threshold, max_length_threshold, verbose=verbose
    )

    mesh_ = subdivide_triangles(mesh, cells_to_divide)

    return mesh_


def subdivide_triangles_on_base_mesh(
    base_mesh,
    mesh,
    area_threshold=None,
    length_threshold=None,
    max_length_threshold=None,
    verbose=False,
):
    """
    Subdivide large triangles in a mesh by splitting specified cells (faces/triangles) on a base mesh.
    The base mesh is usually the original mesh before it was interpolated to be `mesh`. The purpose of
    this is to increase density of vertices/cells in the `base_mesh` so it can preserve details of the
    mesh after the interpolation is completed.

    Parameters:
    - base_mesh: A PyVista mesh.
    - mesh: A PyVista mesh.
    - area_threshold: The maximum area of a triangle before it is subdivided.
    - length_threshold: The maximum:min edge length ratio of a triangle before it is subdivided.
    - max_length_threshold: The maximum edge length of a triangle before it is subdivided.
    - verbose: Whether to print additional information.

    Returns:
    - mesh_: A new PyVista mesh copy of the base_mesh with the specified cells split into 4 sub-triangles.
    and adjacent cells split into 2 sub-triangles.

    """
    cells_to_divide = get_target_cells(
        mesh, area_threshold, length_threshold, max_length_threshold, verbose=verbose
    )

    n_cells = base_mesh.GetNumberOfCells()

    mesh_ = subdivide_triangles(base_mesh, cells_to_divide)

    cell_colors = np.zeros(mesh_.GetNumberOfCells())
    cell_colors[n_cells:] = 1

    cell_colors = numpy_to_vtk(cell_colors)
    cell_colors.SetName("cell_color")
    mesh_.GetCellData().AddArray(cell_colors)

    return mesh_
