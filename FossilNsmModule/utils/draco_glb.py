# GLB (Draco) -> vtkPolyData -> .vtk; no Slicer imports, vtk/numpy/DracoPy loaded lazily.
import base64
import json
import struct

_IDENTITY_MATRIX = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def _parse_glb(path):
    # Return (gltf_json, binary_chunk); handles BIN-chunk and data-URI buffers.
    data = open(path, "rb").read()
    magic, version, _ = struct.unpack("<4sII", data[:12])
    if magic != b"glTF" or version != 2:
        raise ValueError("not a glTF 2.0 binary file: {}".format(path))
    gltf, binary = None, None
    off = 12
    while off < len(data):
        clen, ctype = struct.unpack("<I4s", data[off:off + 8])
        payload = data[off + 8:off + 8 + clen]
        if ctype == b"JSON":
            gltf = json.loads(payload)
        elif ctype.startswith(b"BIN"):
            binary = payload
        off += 8 + clen
    if binary is None and gltf and gltf.get("buffers"):
        uri = gltf["buffers"][0].get("uri", "")
        if uri.startswith("data:"):
            binary = base64.b64decode(uri.split(",", 1)[1])
    return gltf, binary


def _approx(values, target, tol=1e-6):
    return len(values) == len(target) and all(abs(a - b) <= tol for a, b in zip(values, target))


def _check_node_transform(gltf):
    # Refuse a genuinely non-identity node transform; identity transforms are fine.
    # TODO: unvalidated against real data (fixture absent) - confirm against the committed fixture.
    node = next((n for n in gltf.get("nodes", []) if n.get("mesh") == 0), {})
    checks = (
        ("matrix", _IDENTITY_MATRIX),
        ("translation", [0, 0, 0]),
        ("rotation", [0, 0, 0, 1]),
        ("scale", [1, 1, 1]),
    )
    for key, identity in checks:
        if key in node and not _approx(node[key], identity):
            raise NotImplementedError("node carries a non-identity '{}' transform; must be applied".format(key))


def glb_to_polydata(path):
    import numpy as np
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    try:
        import DracoPy
    except ImportError:
        raise ImportError("DracoPy is required to decode Draco .glb meshes. pip_install('DracoPy').")

    gltf, binary = _parse_glb(path)
    if not gltf:
        raise ValueError("glTF has no JSON chunk: {}".format(path))
    meshes = gltf.get("meshes")
    if not meshes or not meshes[0].get("primitives"):
        raise ValueError("glTF has no meshes/primitives: {}".format(path))
    prim = meshes[0]["primitives"][0]
    ext = prim.get("extensions", {}).get("KHR_draco_mesh_compression")
    if ext is None:
        raise NotImplementedError(
            "Uncompressed glTF - decode via accessors instead (not needed for this dataset)")
    if binary is None:
        raise ValueError(
            "glTF buffer is external (.bin) or missing; only self-contained .glb is supported: {}".format(path))

    bv = gltf["bufferViews"][ext["bufferView"]]
    start = bv.get("byteOffset", 0)
    buf = binary[start:start + bv["byteLength"]]

    mesh = DracoPy.decode(buf)
    verts = np.asarray(mesh.points, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)

    _check_node_transform(gltf)

    if verts.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError("decoded mesh is empty (zero vertices or faces): {}".format(path))

    # Zero-extent bounds are the trimesh zero-mesh signature; never render it.
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    if not np.any(hi - lo):
        raise ValueError("decoded mesh has zero-extent bounds (empty/invalid decode): {}".format(path))

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(verts, deep=True))

    cells = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    ca = vtk.vtkCellArray()
    ca.SetCells(len(faces), numpy_to_vtkIdTypeArray(cells, deep=True))

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetPolys(ca)
    return pd


def write_vtk(pd, out_path):
    import vtk
    w = vtk.vtkPolyDataWriter()
    w.SetFileName(out_path)
    w.SetInputData(pd)
    w.SetFileTypeToBinary()
    w.Write()
    return out_path
