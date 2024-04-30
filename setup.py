from setuptools import setup

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
    packages = [ "carbonara", "carbonara.src", "carbonara.model.save" ],
    include_package_data = True,
    package_dir = { "carbonara": "." },
    package_data = {
        "carbonara": [
            "src/*.py",
            "model/save/s_v6_4_2022-09-16_11-51/*.py",
            "model/save/s_v6_4_2022-09-16_11-51/model.pt",
            "model/save/s_v6_4_2022-09-16_11-51/src/*.py",
            "model/save/s_v6_4_2022-09-16_11-51/s_v6_4_2022-09-16_11-51_cdf.csv",
        ],
    },
    entry_points = {
        "console_scripts": [
            "carbonara=carbonara.carbonara:main",
        ],
    },
    install_requires = requirements,
)
