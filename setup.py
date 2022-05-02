from setuptools import setup
from pathlib import Path 
import versioneer

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
setup(  name='bslib',
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(), 
        description='A library for simulating PV battery storage systems.',
        install_requires=['numpy >= 1.16.0', 
                        'pandas >= 0.22.0'],
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/RE-Lab-Projects/bslib',
        author='RE-Lab HS-Emden-Leer',
        packages=['bslib'],
        package_data={'bslib': ['bslib/*.csv']},
        include_package_data=True
        )
