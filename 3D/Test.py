import vtk

# Create points for the grid
points = vtk.vtkPoints()
for x in range(11):
    for y in range(11):
        points.InsertNextPoint(x, y, 0)
        points.InsertNextPoint(x, y, 10)
        points.InsertNextPoint(x, 0, y)
        points.InsertNextPoint(x, 10, y)
        points.InsertNextPoint(0, x, y)
        points.InsertNextPoint(10, x, y)

# Create a polydata to hold the points
polydata = vtk.vtkPolyData()
polydata.SetPoints(points)

# Create a grid lines
lines = vtk.vtkCellArray()

# Add the grid lines
for i in range(11):
    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, i * 11)
    line.GetPointIds().SetId(1, i * 11 + 10)
    lines.InsertNextCell(line)

    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, i)
    line.GetPointIds().SetId(1, 110 + i)
    lines.InsertNextCell(line)

    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, i * 11)
    line.GetPointIds().SetId(1, i)
    lines.InsertNextCell(line)

# Create a polydata to hold the grid lines
gridPolyData = vtk.vtkPolyData()
gridPolyData.SetPoints(polydata.GetPoints())
gridPolyData.SetLines(lines)

# Create mapper and actor for the grid lines
gridMapper = vtk.vtkPolyDataMapper()
gridMapper.SetInputData(gridPolyData)

gridActor = vtk.vtkActor()
gridActor.SetMapper(gridMapper)

# Create renderer, render window, and interactor
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.1, 0.2, 0.4)  # Set background color

renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Add grid actor to the scene
renderer.AddActor(gridActor)

# Set up camera
camera = vtk.vtkCamera()
camera.SetPosition(10, 10, 10)
camera.SetFocalPoint(5, 5, 5)
camera.SetViewUp(0, 1, 0)
renderer.SetActiveCamera(camera)
renderer.ResetCamera()

# Start interaction
renderWindow.Render()
renderWindowInteractor.Start()
