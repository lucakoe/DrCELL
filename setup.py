from setuptools import setup, find_packages

# TODO complete setup.py
# TODO run through black
# TODO sphynx
setup(
    name='drcell',
    version='0.0.1rc0',
    author='Luca K.',
    author_email='pypi@use.startmail.com',
    description='GUI to generate, cluster and optimize dimensionality reduction output',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/lucakoe/DrCELL',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Windows',
    ],
    entry_points={
        'console_scripts': [
            'drcell-app=drcell.scripts.startApplication:main',
            'drcell-server=drcell.scripts.startBokehServer:main',
            'drcell-convert-legacy=drcell.scripts.convertLegacyMatFiles:main',
        ],
    },
    package_data={
        "my_package": ["resources/logo.jpg"],  # Include the icon in the package
    },
    python_requires='>=3.6',
    install_requires=[line.strip() for line in open('requirements.txt') if line.strip() and not line.startswith('#')],
)
