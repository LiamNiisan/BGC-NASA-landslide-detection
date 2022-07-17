from setuptools import setup, find_packages

install_requires = [
    "dateutils",
    "word2number",
    "dateparser",
    "pandas",
    "flair",
    "numpy",
    "sklearn",
    "geocoder",
    "geopy",
    "spacy",
    "gensim",
    "fasttext",
    "psaw",
    "joblib",
    "newspaper3k",
    "nltk",
    "torch",
    "transfomers"
]

dev_requires = [
    "autopep8",
    "black",
    "pip-tools",
]

setup(
    name="BGC-NASA-landslide-detection",
    version="0.0.1",
    author="UBC",
    description="Landslide Project",
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=install_requires,
    python_requires=">=3.9",
    setup_requires=["pytest-runner"],
    tests_require=dev_requires,
    extras_require={"dev": dev_requires},
)
