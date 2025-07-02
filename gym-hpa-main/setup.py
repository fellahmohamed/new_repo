from setuptools import find_packages, setup

setup(name='gym_hpa',
      version='0.0.1',
      author='José Santos',
      packages=find_packages(include=['gym_hpa', 'gym_hpa.*', 'policies', 'policies.*']),
      author_email='josepedro.pereiradossantos@ugent.be',
      install_requires=['gym', 'numpy', 'keras']  # Add dependencies
)