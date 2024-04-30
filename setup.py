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
    py_modules=['carbonara'],
    include_package_data = True,
    packages=find_packages(where="."),
    entry_points = {
        "console_scripts": [
            "carbonara=carbonara:main",
        ],
    },
    install_requires = requirements,
)
