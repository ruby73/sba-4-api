import setuptools

with open("READ<E.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='com_sba_api', 
    version='1.0',
    description='Python Distribution Utilities',
    author='parkeunsol',
    author_email='ouueyeb1@gmail.com',
    url='https://www.python.org/sigs/distutils-sig/', # build 전에 수정할 것 
    packages= setuptools.find_packages(),

)
