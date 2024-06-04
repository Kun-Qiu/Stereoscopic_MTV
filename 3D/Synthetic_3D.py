import vtk
import numpy as np
from PIL import Image


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
    render_window.SetSize(400, 400)

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


# Main function
def main():
    num_lines = 20
    angle = 30  # Angle of the grid lines in degrees

    polydata = create_gridlines(num_lines, angle)
    actor = create_actor(polydata)

    renderer = create_renderer(actor)
    render_window = create_render_window(renderer)

    # Create two cameras, 100 degrees apart
    camera_left = vtk.vtkCamera()
    camera_left.SetPosition(10, 10, 10)
    camera_left.SetFocalPoint(0, 0, 0)
    camera_left.SetViewUp(0, 0, 1)
    camera_left.Azimuth(50)  # Example azimuth angle

    camera_right = vtk.vtkCamera()
    camera_right.SetPosition(10, 10, 10)
    camera_right.SetFocalPoint(0, 0, 0)
    camera_right.SetViewUp(0, 0, 1)
    camera_right.Azimuth(150)  # Example azimuth angle + 100 degrees

    # Render left view
    renderer.SetActiveCamera(camera_left)
    render_window.Render()
    save_rendered_view(render_window, 'left_view.png')

    # Render right view
    renderer.SetActiveCamera(camera_right)
    render_window.Render()
    save_rendered_view(render_window, 'right_view.png')

    # Open the saved images
    left_image = Image.open('left_view.png')
    right_image = Image.open('right_view.png')

    # Create anaglyph image
    anaglyph_image = create_anaglyph(left_image, right_image)
    anaglyph_image.show()


if __name__ == "__main__":
    main()
