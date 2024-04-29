from setuptools import setup, find_packages

requirements = []
with open("requirements.txt", "r") as fs:
    requirements = [line.strip() for line in fs]

setup(
    name = "carbonara",
    description = "CARBonAra: Context-aware geometric deep learning for protein sequence design.",
    author = "Lucien Krapp",
    author_email = "lucienkrapp@gmail.com",
    url = "https://github.com/lfkrapp/CARBonAra",
    version = "1.0.0",
    license = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License",
    include_package_data = True,
    package_data = {
        "": ["*.py", "src/*.py", "requirements.txt", "LICENSE"],
        "model/save/s_v6_4_2022-09-16_11-51": ["*.py", "src/*.py"],
    },
    exclude_package_data = {
        "": ["*.ipynb", "data", "examples", "md_analysis", "model_analysis", "model_comparison", "results"],
    },
    packages=find_packages(where="."),
    entry_points = {
        "console_scripts": [
            "carbonara=carbonara:main",
        ],
    },
    install_requires = requirements,
)
