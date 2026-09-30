import os

from struphy.models import LinearMHD, VariationalBarotropicFluid


def test_every_feec_variable_gets_background_and_perturbation(tmp_path):
    model = LinearMHD()
    path = model.generate_default_parameter_file(path=str(tmp_path / "params.py"), prompt=False)
    with open(path) as f:
        txt = f.read()

    for sn, species in model.species.items():
        for vn in species.variables:
            assert f"model.{sn}.{vn}.add_background(" in txt
            assert f"model.{sn}.{vn}.add_perturbation(" in txt


def test_relative_path_in_new_folder_stays_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    VariationalBarotropicFluid().generate_default_parameter_file(path="new_folder/params.py", prompt=False)
    assert os.path.isfile(tmp_path / "new_folder" / "params.py")


def test_variational_barotropic_fluid_keeps_barotropic_option(tmp_path):
    path = str(tmp_path / "params.py")
    VariationalBarotropicFluid().generate_default_parameter_file(path=path, prompt=False)
    with open(path) as f:
        txt = f.read()

    assert txt.count("model.propagators.variat_dens.options =") == 1
    assert "variat_dens.Options(model='barotropic')" in txt
    assert txt.count("model.fluid.density.add_background(") == 1
