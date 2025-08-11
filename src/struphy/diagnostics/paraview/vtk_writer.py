import os

import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import numpy_to_vtk as np2vtk
from vtkmodules.util.numpy_support import vtk_to_numpy as vtk2np
from vtkmodules.vtkIOXML import vtkXMLRectilinearGridWriter, vtkXMLUnstructuredGridWriter

"""
Some useful resources:  
- https://vtk.org/Wiki/VTK/Tutorials/DataStorage  
- https://stackoverflow.com/questions/59301207/save-and-write-a-vtk-polydata-file  
- https://stackoverflow.com/questions/54603267/how-to-show-vtkunstructuredgrid-in-python-script-based-on-paraview  
- https://kitware.github.io/vtk-examples/site/Python/  
- https://kitware.github.io/vtk-examples/site/Python/UnstructuredGrid/UGrid/  
- https://github.com/Kitware/VTK/tree/master/Wrapping/Python  
- https://pypi.org/project/meshio/  
- https://stackoverflow.com/questions/59651524/writing-vtk-file-from-python-for-use-in-paraview  
- https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf  

Outdated resources:  
- https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python  
- https://github.com/pearu/pyvtk  
- https://github.com/paulo-herrera/PyEVTK  
- https://shocksolution.com/microfluidics-and-biotechnology/visualization/python-vtk-paraview/  
"""


class vtkWriter:
    """Usage `from struphy.io.out.paraview.vtk_writer import vtkWriter`"""

    def __init__(self, format: str = "vtu"):
        """Initialize vtkWriter object.

        Parameters
        ----------
        format : str
            Type of XML data that is to be written, denoted by its file extension.
        """

        self.format = format

        # Writes .vtu
        self.vtu_writer = vtkXMLUnstructuredGridWriter()

        # Writes .vtr
        self.vtr_writer = vtkXMLRectilinearGridWriter()

        # Writes .vti
        self.vti_writer = vtk.vtkXMLImageDataWriter()

        # Writes .vtp
        self.vtp_writer = vtk.vtkXMLPolyDataWriter()

        if format == "vtu":
            self.writer = self.vtu_writer
        elif format == "vtr":
            self.writer = self.vtr_writer
        # Others not implemented.
        else:
            raise NotImplementedError(".{} ParaView file format not implemented.".format(format))

    def write(self, directory: str, filename: str, ugrid):
        """Write the `vtkUnstructuredGrid` object into a `.vtu` file.

        Parameters
        ----------
        directory : str
            Output directory.
        filename : str
            Output filename, WITHOUT file extension.
        ugrid : vtk.vtkUnstructuredGrid
            A `vtkUnstructuredGrid` object.

        Returns
        -------
        success : bool
            Whether file write is successful.
        """

        writer = self.writer
        writer.SetInputDataObject(ugrid)

        filepath = os.path.join(directory, filename + "." + writer.GetDefaultFileExtension())
        os.makedirs(directory, exist_ok=True)  # Make sure directory exists.
        writer.SetFileName(filepath)
        success = writer.Write()

        return success == 1
