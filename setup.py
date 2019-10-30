from setuptools import setup

setup(name='matc',
      version='0.0.0',
      description='matrix completion package',
      url='https://github.com/zhengp0/MatrixCompletion',
      author='Marlena Bannick, Peng Zheng',
      author_email='mnorwood@uw.edu, zhengp@uw.edu',
      license='MIT',
      packages=['matc'],
      package_dir={'matc': 'src/matc'},
      install_requires=['numpy', 'pytest'],
      zip_safe=False)
