import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description= f.read()

__version__= "0.0.0"

REPO_NAME= "end-to-end_mazie_disease_classification"
AUTHOR_USER_NAME="AbdulRasheed6"
SRC_REPO="cnnClassifier"
AUTHOR_EMAIL= "abdulrasheedolakiitan@gmail.com"


setuptools.setup(
    name= SRC_REPO,
    author= AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A python package for a disease detection app",
    long_description=long_description,
    long_description_conent="text/markdown",
    url= f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker":f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues" 
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)