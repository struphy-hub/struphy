import os
import subprocess

from setuptools import setup
from setuptools.command.install import install


class CustomInstall(install):
    def run(self):
        wheel_path = os.path.join(
            os.path.dirname(__file__),
            "src/struphy/psydac-2.6.0.dev0-py3-none-any.whl",
        )

        subprocess.run(["pip", "install", wheel_path])
        super().run()


setup(
    cmdclass={"install": CustomInstall},
)
