def test_paraview(sim_path=None):
    """Test writing the default simulation output to ParaView."""

    # ============================================================
    # Imports.
    # ============================================================

    import numpy as np

    import os
    import sys
    import sysconfig

    import h5py
    import yaml
    import tempfile
    temp_dir = tempfile.TemporaryDirectory(prefix='STRUPHY-')
    print(f'Created temp directory at: {temp_dir.name}')

    # which diagnostics is run
    print('Run diagnostics:', sys.argv[0])

    basedir = os.path.dirname(os.path.realpath(__file__))
    # sys.path.insert(0, os.path.join(basedir, '..'))

    # Import necessary struphy.modules.
    import struphy.geometry.domain_3d as dom
    import struphy.feec.spline_space as spl

    # Import ParaView-related modules.
    import vtkmodules.all as vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy as vtk2np
    from vtkmodules.util.numpy_support import numpy_to_vtk as np2vtk
    from vtkmodules.vtkCommonCore import vtkDoubleArray
    from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridWriter
    # from vtkmodules.vtkIOParallelXML
    from struphy.diagnostics.paraview.vtk_writer import vtkWriter
    import struphy.diagnostics.paraview.mesh_creator as MC
    import xml.etree.ElementTree as ET
    from xml.dom import minidom



    # ============================================================
    # Load simultion data.
    # ============================================================

    # Obtain path to default simulation output folder.
    # A sample simulation has to be executed by calling `struphy` standalone in bash.
    if sim_path is None:
        platlib = sysconfig.get_path('platlib')
        struphy_path = os.path.join(platlib, 'struphy')
        sim_path = os.path.join(struphy_path, 'io', 'out', 'sim_1')
        print('Identifying default simulation output paths:')
        print(f'platlib     : {platlib}')
        print(f'struphy_path: {struphy_path}')
        print(f'sim_path    : {sim_path}')
    data_path = os.path.join(sim_path, 'data.hdf5')
    params_path = os.path.join(sim_path, 'parameters.yml')
    print(f'data_path   : {data_path}')
    print(f'params_path : {params_path}')

    # Confirm that simulation path exists before moving on:
    if not os.path.isdir(sim_path):
        print(f'Simulation directory does not exist. Aborting...')
        print(f'sim_path    : {sim_path}')
        return
    if not os.path.isfile(data_path):
        print(f'Simulation data does not exist. Aborting...')
        print(f'data_path   : {data_path}')
        return
    if not os.path.isfile(params_path):
        print(f'Simulation parameter file does not exist. Aborting...')
        print(f'params_path : {params_path}')
        return

    # Load params and data.
    data = h5py.File(data_path, 'r')
    keys = list(data.keys())
    print(f'The data file contains the following keys:')
    print(keys)

    with open(params_path) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)



    # ============================================================
    # Reconstruct STRUPHY's spline representation of the output.
    # ============================================================

    Nel      = params['grid']['Nel']      # Number of elements.
    p        = params['grid']['p']        # Spline degree.
    spl_kind = params['grid']['spl_kind'] # Periodic or not.
    bc       = params['grid']['bc']       # BC in s-direction.
    nq_el    = params['grid']['nq_el']    # Element integration.
    nq_pr    = params['grid']['nq_pr']    # Greville integration (histopolation).
    polar    = params['grid']['polar']    # Use polar spline or not.



    # ============================================================
    # Create FEM space and setup projectors.
    # ============================================================

    # 1D B-spline spline spaces for finite elements.
    spaces = [spl.Spline_space_1d(Nel, p, spl_kind, nq_el, bc) for Nel, p, spl_kind, nq_el in zip(Nel, p, spl_kind, nq_el)]
    [space.set_projectors(nq=nq) for space, nq in zip(spaces, nq_pr) if not hasattr(space, 'projectors')]

    # 3D tensor-product B-spline spline space for finite elements.
    TENSOR_SPACE = spl.Tensor_spline_space(spaces, ck=-1)
    if not hasattr(TENSOR_SPACE, 'projectors'):
        TENSOR_SPACE.set_projectors('general') # def set_projectors(self, which='tensor'). Use 'general' for polar splines.
    print('Tensor space and projector set up done.')



    # ============================================================
    # Parse simulation data (spline coefficients).
    # ============================================================

    for variable in keys:
        print(f'{variable} has shape {data[variable].shape}')

    time = data['time'][:]
    # print(f'Time: {time}') # Time is a zero array!

    parsed = {}

    for variable in keys:

        if variable in ['density', 'pressure']: # 3-form
            parsed[variable] = []
            for t in range(len(time)):
                temp = TENSOR_SPACE.extract_3(data[variable][:][t])
                parsed[variable].append(temp)
            parsed[variable] = np.array(parsed[variable])
            print(f'Parsed {variable} has shape {parsed[variable].shape}')

        elif variable in ['magnetic field', 'mhd velocity']: # 2-form
            parsed[variable] = []
            for t in range(len(time)):
                temp = TENSOR_SPACE.extract_2(data[variable][:][t])
                parsed[variable].append(temp)
            parsed[variable] = np.array(parsed[variable])
            print(f'Parsed {variable} has shape {parsed[variable].shape}')



    # ============================================================
    # Create domain.
    # ============================================================

    geometry = params['geometry']['type']
    params_map = params['geometry'][geometry]
    DOMAIN = dom.Domain(geometry, params_map)
    print(f'Geometry: {geometry}')
    print(f'Params map: {params_map}')



    # ============================================================
    # Evaluate at points of interest (vtk mesh) and push.
    # ============================================================

    eta1_range = np.linspace(1e-8, 1, 21)
    eta2_range = np.linspace(0, 1, 23)
    eta3_range = np.linspace(0, 1, 25)

    # Use spline element boundaries for grid.
    eta1_range, eta2_range, eta3_range = [np.array(space.el_b) for space in TENSOR_SPACE.spaces]

    len1, len2, len3 = eta1_range.shape[0], eta2_range.shape[0], eta3_range.shape[0]
    eta1, eta2, eta3 = np.meshgrid(eta1_range, eta2_range, eta3_range, indexing='ij', sparse=False)

    # Need to be careful if DOMAIN is time-dependent. Not handled.
    x = DOMAIN.evaluate(eta1, eta2, eta3, 'x')
    y = DOMAIN.evaluate(eta1, eta2, eta3, 'y')
    z = DOMAIN.evaluate(eta1, eta2, eta3, 'z')

    pushed = {}

    for variable in keys:

        if variable in ['density', 'pressure']: # 3-form
            pushed[variable] = []
            for t in range(len(time)):
                evaled_3 = TENSOR_SPACE.evaluate_DDD(eta1, eta2, eta3, parsed[variable][t])
                pushed_3 = DOMAIN.push(evaled_3, eta1, eta2, eta3, kind_fun='3_form')
                pushed[variable].append(pushed_3)
            pushed[variable] = np.array(pushed[variable])
            print(f'Pushed {variable} has shape {pushed[variable].shape}')

        elif variable in ['magnetic field', 'mhd velocity']: # 2-form
            pushed[variable] = []
            for t in range(len(time)):
                evaled_2_1 = TENSOR_SPACE.evaluate_NDD(eta1, eta2, eta3, parsed[variable][t][0])
                evaled_2_2 = TENSOR_SPACE.evaluate_DND(eta1, eta2, eta3, parsed[variable][t][1])
                evaled_2_3 = TENSOR_SPACE.evaluate_DDN(eta1, eta2, eta3, parsed[variable][t][2])
                evaled_2   = [evaled_2_1, evaled_2_2, evaled_2_3]
                pushed_2_1 = DOMAIN.push(evaled_2, eta1, eta2, eta3, kind_fun='2_form_1')
                pushed_2_2 = DOMAIN.push(evaled_2, eta1, eta2, eta3, kind_fun='2_form_2')
                pushed_2_3 = DOMAIN.push(evaled_2, eta1, eta2, eta3, kind_fun='2_form_3')
                pushed_2   = np.array([pushed_2_1, pushed_2_2, pushed_2_3])
                pushed[variable].append(pushed_2)
            pushed[variable] = np.array(pushed[variable])
            print(f'Pushed {variable} has shape {pushed[variable].shape}')



    temp_dir.cleanup()
    print('Removed temp directory.')



    # ============================================================
    # Associate evaluated data with mesh vertices.
    # ============================================================

    num_pts = len1 * len2 * len3
    print(f'Number of points: {num_pts}', flush=True)

    cell_data = []
    point_data = []
    vtk_points = []
    point_indices = []

    for t in range(len(time)):

        cell_data.append({})
        point_data.append({})
        vtk_points.append(vtk.vtkPoints())
        point_indices.append(np.zeros((len1, len2, len3), dtype=np.int_))

        point_data[t]['Point ID'] = np.zeros(num_pts, dtype=np.int_)
        point_data[t]['eta1'] = np.zeros(num_pts, dtype=np.float_)
        point_data[t]['eta2'] = np.zeros(num_pts, dtype=np.float_)
        point_data[t]['eta3'] = np.zeros(num_pts, dtype=np.float_)
        point_data[t]['x'] = np.zeros(num_pts, dtype=np.float_)
        point_data[t]['y'] = np.zeros(num_pts, dtype=np.float_)
        point_data[t]['z'] = np.zeros(num_pts, dtype=np.float_)
        point_data[t]['density'] = np.zeros(num_pts, dtype=np.float_)
        point_data[t]['pressure'] = np.zeros(num_pts, dtype=np.float_)
        point_data[t]['mhd velocity'] = np.zeros((num_pts, 3), dtype=np.float_)
        point_data[t]['magnetic field'] = np.zeros((num_pts, 3), dtype=np.float_)

        pt_idx = 0
        # pbar = tqdm(total=num_pts)
        # The x-direction has to be in the inner loop. Then y, then z.
        for idx3, e3 in enumerate(eta3_range):
            for idx2, e2 in enumerate(eta2_range):
                for idx1, e1 in enumerate(eta1_range):
                    point_indices[t][idx1, idx2, idx3] = pt_idx
                    point_data[t]['Point ID'][pt_idx] = pt_idx
                    point_data[t]['eta1'][pt_idx] = e1
                    point_data[t]['eta2'][pt_idx] = e2
                    point_data[t]['eta3'][pt_idx] = e3
                    point_data[t]['x'][pt_idx] = x[idx1, idx2, idx3]
                    point_data[t]['y'][pt_idx] = y[idx1, idx2, idx3]
                    point_data[t]['z'][pt_idx] = z[idx1, idx2, idx3]
                    point_data[t]['density'][pt_idx]        = pushed['density'][t, idx1, idx2, idx3]
                    point_data[t]['pressure'][pt_idx]       = pushed['pressure'][t, idx1, idx2, idx3]
                    point_data[t]['mhd velocity'][pt_idx]   = pushed['mhd velocity'][t, :, idx1, idx2, idx3]
                    point_data[t]['magnetic field'][pt_idx] = pushed['magnetic field'][t, :, idx1, idx2, idx3]
                    vtk_points[t].InsertPoint(pt_idx, [x[idx1, idx2, idx3], y[idx1, idx2, idx3], z[idx1, idx2, idx3]])
                    # pbar.update(1)
                    pt_idx += 1
        # pbar.close()



    # ============================================================
    # Assign data to various VTK grid setups.
    # ============================================================

    grids = []

    for t in range(len(time)):

        # Source: https://kitware.github.io/vtk-examples/site/Python/RectilinearGrid/RGrid/
        if geometry == 'cuboid':

            # Create a rectilinear grid by defining three arrays specifying the coordinates in the x-y-z directions.
            xCoords = vtkDoubleArray()
            for i in range(len1):
                xCoords.InsertNextValue(x[i,0,0])
            yCoords = vtkDoubleArray()
            for i in range(len2):
                yCoords.InsertNextValue(y[0,i,0])
            zCoords = vtkDoubleArray()
            for i in range(len3):
                zCoords.InsertNextValue(z[0,0,i])

            # The coordinates are assigned to the rectilinear grid.
            # Make sure that the number of values in each of the XCoordinates, YCoordinates, and ZCoordinates is equal to what is defined in SetDimensions().
            grid = vtkRectilinearGrid()
            grid.SetDimensions(len1, len2, len3)
            grid.SetXCoordinates(xCoords)
            grid.SetYCoordinates(yCoords)
            grid.SetZCoordinates(zCoords)
            print(f'There are {grid.GetNumberOfPoints()} points.')
            print(f'There are {grid.GetNumberOfCells()} cells.')

            # https://vtk.org/doc/nightly/html/classvtkRectilinearGrid.html#a91fcb6b3e1438a1a01f8ddf3ebc8937a
            # "Given a user-supplied vtkPoints container object, this method fills in all the points of the RectilinearGrid."
            # Then why isn't this called SetPoints!?
            grid.GetPoints(vtk_points[t])

        else:

            grid = vtk.vtkUnstructuredGrid()
            grid.SetPoints(vtk_points)
            grid.Allocate(num_pts)
            MC.connect_cell(eta1_range, eta2_range, eta3_range, point_indices[t], grid, point_data[t], cell_data[t], periodic=[False,False,False])

        # Getting the VTK data storage object.
        vtk_point_data = grid.GetPointData()
        vtk_cell_data  = grid.GetCellData()

        # For each numpy data array, convert it into VTK format, set its name, and add to grid.
        for i, (k, v) in enumerate(point_data[t].items()):
            vtk_array = np2vtk(v)
            vtk_array.SetName(k)
            vtk_point_data.AddArray(vtk_array)

        for i, (k, v) in enumerate(cell_data[t].items()):
            vtk_array = np2vtk(v)
            vtk_array.SetName(k)
            vtk_cell_data.AddArray(vtk_array)

        grids.append(grid)



    # ============================================================
    # Write to ParaView.
    # ============================================================

    print(f'Writing result to ParaView')
    print(f'VTK version: {vtk.vtkVersion.GetVTKVersion()}')

    # Output directory.
    vtk_dir = os.path.join(basedir, 'paraview_output')
    filename = 'TestTimeSeq'

    # Create XML-based PVD file.
    pvd_root = ET.Element('VTKFile', type='Collection')
    collection = ET.SubElement(pvd_root, 'Collection')

    if geometry == 'cuboid':

        writer = vtkXMLRectilinearGridWriter()

    else:

        # Class implementation of a ParaView writer.
        writer = vtkWriter('vtu').writer # Unstructured grid writer.

    digits = int(np.log10(len(time)))+1
    fmt = f'0{digits}d'

    for i, t in enumerate(time):
        writer.SetInputDataObject(grid)
        filepath = os.path.join(vtk_dir, filename + f'_{i:{fmt}}.' + writer.GetDefaultFileExtension())
        # TODO: Should use actual timestep for attribute `timestep=f'{t}'`, but currently it is always zero. Use index `i` instead.
        ET.SubElement(collection, 'DataSet', timestep=f'{i}', part='0', file=filename + f'_{i:{fmt}}.' + writer.GetDefaultFileExtension())
        os.makedirs(vtk_dir, exist_ok=True) # Make sure directory exists.
        writer.SetFileName(filepath)
        success = writer.Write()
        print(f'Success writing ParaView file for the {i:{fmt}}-th timestep: {success==1}.')

    filepath = os.path.join(vtk_dir, f'{filename}.pvd')
    # tree = ET.ElementTree(pvd_root)
    # tree.write(filepath, encoding='UTF-8', xml_declaration=True)
    # To pretty-print xml:
    xmlstr = minidom.parseString(ET.tostring(pvd_root)).toprettyxml(indent='    ')
    with open(filepath, 'w') as f:
        f.write(xmlstr)



if __name__ == "__main__":
    test_paraview()
