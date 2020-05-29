from setuptools import setup, find_packages

setup(name='gzbuilder_analysis',
      version='0.1',
      description='Model optimization package for Galaxy Zoo: Builder',
      author='Tim Lingard',
      author_email='tklingard@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'scikit-learn',
          'numba',
          'shapely',
          'tqdm',
          'montage-wrapper',
          'sep',
          'astropy',
          'requests',
          'Pillow',
      ],
      zip_safe=False)
