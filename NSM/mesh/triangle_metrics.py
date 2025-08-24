import vtk
import numpy as np


def get_triangle_area(cell):
    if cell.GetCellType() == vtk.VTK_TRIANGLE:
        p0 = cell.GetPoints().GetPoint(0)
        p1 = cell.GetPoints().GetPoint(1)
        p2 = cell.GetPoints().GetPoint(2)

        # Compute the area using vtkTriangle
        area = vtk.vtkTriangle.TriangleArea(p0, p1, p2)
    else:
        raise Exception("only support triangle meshes")

    return area


def calculate_triangle_areas(polyData):
    areas = []
    for i in range(polyData.GetNumberOfCells()):
        cell = polyData.GetCell(i)
        area = get_triangle_area(cell)
        areas.append(area)
    return areas


def length(x1, x2):
    return np.sqrt(sum((x1 - x2) ** 2))


def get_edge_lengths(cell):
    p0 = np.asarray(cell.GetPoints().GetPoint(0))
    p1 = np.asarray(cell.GetPoints().GetPoint(1))
    p2 = np.asarray(cell.GetPoints().GetPoint(2))

    edge_lengths = []
    edge_lengths.append(length(p0, p1))
    edge_lengths.append(length(p1, p2))
    edge_lengths.append(length(p2, p0))

    return edge_lengths


class TriangleProperties:
    def __init__(self, mesh):
        self._mesh = mesh
        self.edge_lengths = None
        self._areas = None

    def areas(self, norm=True):
        if self._areas is None:
            self._areas = calculate_triangle_areas(self._mesh)

        if norm is True:
            ref_area = np.mean(self._areas)
            areas = (self._areas - ref_area) / ref_area
        else:
            areas = self._areas.copy()

        return np.asarray(areas)

    def compute_edge_lengths(self):
        self.edge_lengths = []
        for i in range(self._mesh.GetNumberOfCells()):
            cell = self._mesh.GetCell(i)
            lengths = get_edge_lengths(cell)
            self.edge_lengths.append(lengths)

        self.edge_lengths = np.array(self.edge_lengths)

    def edge_ratio(self):
        if self.edge_lengths is None:
            self.compute_edge_lengths()

        min_ = np.min(self.edge_lengths, axis=1)
        max_ = np.max(self.edge_lengths, axis=1)

        if sum(min_ == 0) > 0:
            zero_area = np.where(min_ == 0)
            raise Exception(f"edge length zero! triangle with zero length edge: {zero_area}")

        lengths_ratio = max_ / min_

        return lengths_ratio

    def edge_sd(self):
        if self.edge_lengths is None:
            self.compute_edge_lengths()

        return np.std(self.edge_lengths, axis=1)

    def edge_length_max(self):
        if self.edge_lengths is None:
            self.compute_edge_lengths()

        return np.max(self.edge_lengths, axis=1)
