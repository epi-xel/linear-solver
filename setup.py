from setuptools import setup, find_packages

setup(
    name='linear-system-solver',
    version='1.0',
    packages=find_packages(),
    package_data={'test': ['*.mtx']},
    include_package_data=True,
    install_requires=[
        # lista delle dipendenze del programma
    ],
    entry_points={
        'console_scripts': [
            'nome_comando=nome_modulo:funzione_principale',
        ],
    },
)
