import os
from setuptools import setup, find_packages

# Utility function to read the README file.
def read(fname):
    try:
        with open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8') as f:
            content = f.read()
            return content
    except FileNotFoundError:
        return "Generating Vocabulary Management Profiles in Python"

setup(
    name="vmp",
    version="0.0.4",  # Updated version number
    author="Matthew Durward, Christopher Thomson",
    author_email="matthew.durward@pg.canterbury.ac.nz",
    description="Generating Vocabulary Management Profiles in Python",
    package_dir={"": "vmp"},
    packages=find_packages(where="vmp"),
    license="GNU GENERAL PUBLIC LICENSE v3",
    url="https://github.com/matthewdurward/vmp",
    keywords="text analytics, natural language processing, computational linguistics, vocabulary, lexical diversity, corpus, corpora",
    install_requires=[
        'pandas',
        'tqdm',
        'clean-text',
        'regex',
        'requests',
    ],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    include_package_data=True,
    long_description=read('Readme.md'),
    long_description_content_type='text/markdown',
)
