from setuptools import setup
setup(
    name='benchmark',
    version='1.0.0',    
    description='A simple Python wrapper to benchmark TensorFlow and PyTorch models.',
    author='Federico Ricciuti',
    author_email='ricciuti.federico@gmail.com',
    packages=['benchmark'],
    package_data={'benchmark': [
        'examples/', 
    ]}
)