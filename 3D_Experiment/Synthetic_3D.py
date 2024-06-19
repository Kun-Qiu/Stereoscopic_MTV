import vtk
import numpy as np
from PIL import Image
import math


def calculate_camera_position(focal_point, view_angle, distance):
    angle_radians = math.radians(view_angle)
    camera_x = distance * math.cos(angle_radians / 2)
    camera_y = distance * math.sin(angle_radians / 2)
    camera_position = [focal_point[0] + camera_x, focal_point[1] - camera_y, focal_point[2]]
    return camera_position


def create_gridlines(num_lines, angle):
    lines = vtk.vtkCellArray()
    points = vtk.vtkPoints()

    angle_rad = np.radians(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    for i in range(num_lines):
        offset = -10 + 20.0 * i / (num_lines - 1)

        # Line along the specified angle
        x1 = offset * cos_angle - 10 * sin_angle
        y1 = offset * sin_angle + 10 * cos_angle
        x2 = offset * cos_angle + 10 * sin_angle
        y2 = offset * sin_angle - 10 * cos_angle

        id1 = points.InsertNextPoint(x1, y1, 0)
        id2 = points.InsertNextPoint(x2, y2, 0)
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, id1)
        line.GetPointIds().SetId(1, id2)
        lines.InsertNextCell(line)

        # Perpendicular line
        x1_perp = offset * sin_angle - 10 * cos_angle
        y1_perp = -offset * cos_angle - 10 * sin_angle
        x2_perp = offset * sin_angle + 10 * cos_angle
        y2_perp = -offset * cos_angle + 10 * sin_angle

        id1_perp = points.InsertNextPoint(x1_perp, y1_perp, 0)
        id2_perp = points.InsertNextPoint(x2_perp, y2_perp, 0)
        line_perp = vtk.vtkLine()
        line_perp.GetPointIds().SetId(0, id1_perp)
        line_perp.GetPointIds().SetId(1, id2_perp)
        lines.InsertNextCell(line_perp)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    return polydata


def find_intersections(polydata):
    points = polydata.GetPoints()
    num_points = points.GetNumberOfPoints()
    intersections = vtk.vtkPoints()

    for i in range(0, num_points, 4):
        for j in range(i + 4, num_points, 4):
            x1, y1, _ = points.GetPoint(i)
            x2, y2, _ = points.GetPoint(i + 1)
            x3, y3, _ = points.GetPoint(j)
            x4, y4, _ = points.GetPoint(j + 1)

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom != 0:
                px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

                if min(x1, x2) <= px <= max(x1, x2) \
                        and min(y1, y2) <= py <= max(y1, y2) \
                        and min(x3, x4) <= px <= max(x3, x4) \
                        and min(y3, y4) <= py <= max(y3, y4):
                    intersections.InsertNextPoint(px, py, 0)

    return intersections


def create_actor(polydata):
    # Create a mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    # Create an actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(2)
    actor.GetProperty().SetColor(0, 0, 1)  # Set color to blue

    return actor


def create_intersection_actor(points):
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    vertex_glyph_filter = vtk.vtkVertexGlyphFilter()
    vertex_glyph_filter.SetInputData(polydata)
    vertex_glyph_filter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(vertex_glyph_filter.GetOutput())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1, 0, 0)  # Set color to red
    actor.GetProperty().SetPointSize(5)  # Set point size

    return actor


def create_renderer(actor):
    # Create a renderer and add the actor
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)  # Set background color to white

    return renderer


def create_render_window(renderer):
    # Create a render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(512, 512)

    return render_window


def save_rendered_view(render_window, filename):
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.Write()


def create_anaglyph(left_image, right_image):
    left_array = np.array(left_image)
    right_array = np.array(right_image)

    anaglyph_array = np.zeros_like(left_array)
    anaglyph_array[..., 0] = left_array[..., 0]
    anaglyph_array[..., 1] = right_array[..., 1]
    anaglyph_array[..., 2] = right_array[..., 2]

    anaglyph_image = Image.fromarray(anaglyph_array)
    return anaglyph_image


def apply_velocity_field_to_points(points, velocity_field):
    num_points = points.GetNumberOfPoints()
    for i in range(num_points):
        x, y, z = points.GetPoint(i)
        # Assuming velocity field is a function vel(x, y, z) -> (vx, vy, vz)
        vx, vy, vz = velocity_field(x, y, z)
        points.SetPoint(i, x + vx, y + vy, z + vz)


def velocity_field(x, y, z):
    # Simple example: shift everything by 5 units in the x and y direction
    vx = 5
    vy = 5
    vz = 0
    return vx, vy, vz


# Main function
def main():
    num_lines = 11
    angle = 45  # Angle of the grid lines in degrees
    focal_point = [0, 0, 0]
    view_angle = 50
    distance_to_focal = 10

    polydata = create_gridlines(num_lines, angle)
    points = polydata.GetPoints()

    # Find intersections
    intersection_points = find_intersections(polydata)

    # Create actors
    grid_actor = create_actor(polydata)
    intersection_actor = create_intersection_actor(intersection_points)

    # Create renderer and render window
    renderer = create_renderer(grid_actor)
    renderer.AddActor(intersection_actor)
    render_window = create_render_window(renderer)

    camera_left_position = calculate_camera_position(focal_point, view_angle, distance_to_focal)
    camera_right_position = calculate_camera_position(focal_point, view_angle, distance_to_focal)

    # Create cameras
    camera_left = vtk.vtkCamera()
    camera_left.SetPosition(camera_left_position)
    camera_left.SetFocalPoint(focal_point)
    camera_left.SetViewUp(0, 1, 0)
    camera_left.SetViewAngle(view_angle)

    camera_right = vtk.vtkCamera()
    camera_right.SetPosition(camera_right_position)
    camera_right.SetFocalPoint(focal_point)
    camera_right.SetViewUp(0, 1, 0)
    camera_right.SetViewAngle(view_angle)

    # Render left and right views before applying the velocity field
    renderer.SetActiveCamera(camera_left)
    render_window.Render()
    save_rendered_view(render_window, 'left_view_before.png')

    renderer.SetActiveCamera(camera_right)
    render_window.Render()
    save_rendered_view(render_window, 'right_view_before.png')

    # Apply velocity field to the points
    apply_velocity_field_to_points(points, velocity_field)

    # Render left and right views after applying the velocity field
    renderer.SetActiveCamera(camera_left)
    render_window.Render()
    save_rendered_view(render_window, 'left_view_after.png')

    renderer.SetActiveCamera(camera_right)
    render_window.Render()
    save_rendered_view(render_window, 'right_view_after.png')

    # Open the saved images
    left_image_before = Image.open('left_view_before.png')
    right_image_before = Image.open('right_view_before.png')
    left_image_after = Image.open('left_view_after.png')
    right_image_after = Image.open('right_view_after.png')

    # Create anaglyph images
    anaglyph_image_before = create_anaglyph(left_image_before, right_image_before)
    anaglyph_image_after = create_anaglyph(left_image_after, right_image_after)

    # Show anaglyph images
    anaglyph_image_before.show()
    anaglyph_image_after.show()


if __name__ == "__main__":
    main()
