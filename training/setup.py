import os
import re

from setuptools import setup

regexp = re.compile(r".*__version__ = [\'\"](.*?)[\'\"]", re.S)

base_package = "forvia-oms"
base_path = os.path.dirname(__file__)

init_file = os.path.join(base_path, "src", "__init__.py")
with open(init_file, "r") as f:
    module_content = f.read()

    match = regexp.match(module_content)
    if match:
        version = match.group(1)
    else:
        raise RuntimeError("Cannot find __version__ in {}".format(init_file))

with open("README.md", "r") as f:
    readme = f.read()

with open("./docs/CHANGELOG.md", "r") as f:
    changes = f.read()


def parse_requirements(filename):
    """Load requirements from a pip requirements file"""
    with open(filename, "r") as fd:
        lines = []
        for line in fd:
            line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


requirements = parse_requirements("docker/requirements.txt")

if __name__ == "__main__":
    setup(
        name=base_package,
        description="python code for quality evaluation by vision",
        long_description="\n\n".join([readme, changes]),
        license="Not open source",
        url="",
        version=version,
        author="probayes",
        author_email="",
        maintainer="probayes",
        maintainer_email="",
        install_requires=requirements,
        keywords=["forvia", "oms"],
        package_dir={"": "."},
        packages=["src"],
        zip_safe=False,
        classifiers=[
            "Development Status :: 1 - Alpha",
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
    )
