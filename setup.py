from setuptools import setup

setup(name='gzbuilder_analysis',
      version='0.1',
      description='Model optimization package for Galaxy Zoo: Builder',
      author='Tim Lingard',
      author_email='tklingard@gmail.com',
      license='MIT',
      packages=['gzbuilder_analysis'],
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'scikit-learn',
          'numba',
          'shapely',
          'tqdm',
      ],
      zip_safe=False)
