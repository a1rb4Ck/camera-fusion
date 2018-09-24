"""camera_fusion package installation parameters."""

from setuptools import setup
from setuptools import find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='camera-fusion',
    version='0.0.4',
    author='Pierre Nagorny',
    author_email='pierre.nagorny@univ-smb.fr',
    description='Multiple cameras calibration and fusion with OpenCV Python.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/a1rb4Ck/camera-fusion',
    license='MIT',
    install_requires=['numpy>=1.15.1',
                      'opencv-contrib-python >= 3.4.3.18',
                      'sortednp>=0.2.0'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Multimedia :: Video',
        'Topic :: Multimedia :: Video :: Capture'],
    keywords='opencv fusion image video',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    scripts=['bin/camera_calibration', 'bin/cameras_fusion'],
    zip_safe=True)
