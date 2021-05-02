import os
import setuptools
from setuptools.command.build_ext import build_ext

with open("README.md",'r') as f:
	long_description = f.read()

packages = setuptools.find_packages()
requirements = [line.strip() for line in open("./requirements.txt",'r')]

setuptools.setup(
	name="autoagents",
	version="0.0.1",
	author="SISL",
	author_email="sman64@stanford.edu",
	description="A distributed framework for training RL and MPC algorithms on various environments such as OpenAI Gym, and car racing Environments",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/sisl/AutomotiveAgents",
	install_requires=requirements,
	packages_dir="./",
	packages=packages,
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent"
	],
	python_requires=">=3.6",
	cmdclass={}
)