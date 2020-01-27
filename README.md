# Example Python module

## Setup and install

It is best to create a virtual environment for every separate repository. You can create a virtual environment called "env" by executing the following command in the root directory of your repository.
```shell
python -m venv env
```
Now each time you want to use this environment, you need to first activate it with (on Windows):
```shell
./env/Scripts/activate
```
After creating a new virtual environment, it is always a good idea to update pip.
```shell
python -m pip install --upgrade pip
```

In your repository, you should always create a file "requirements.txt" that contains all the required modules (dependencies) using:
```shell
pip freeze > requirements.txt
```

When setting up the repository on a new location, you can use this file to install all required modules using (first activate the virtual environment):
```shell
pip install -r requirements.txt
```

## Running the module

To run this example module you need a configuration JSON. The defualt name is "config.json". Different commands can be implemented that can be executed by running:
```shell
python main.py example_command
```
If the configuration file has a different name then use:
```shell
python main.py --config 'different_name.json' example_command
```

## Pylint
Please use pylint to style check and error check your code. It is one of the installed packages.

## Sample config.json
The JSON files should not be checked in as source code. Therefore, please always include a sample JSON for all functionality included in the module. For this example module, the following JSON can be used.
```json
{
    "message": {
        "number": 3,
        "text": "hello world!"
    }
}
```

## Pip module
To install this module directly from git, you need to update the "setup.py" file.
```python
from setuptools import setup

setup(
    name='Example module',
    version='0.1',
    description='A repository containing an example module',
    author='Nico van Dijk @ Slimstock',
    author_email='n.vandijk@slimstock.com',
    packages=['example_package'],
    url='git@gitlab.com:slimstock-research/example_module.git'
)
```
You can then install this package (in another repository) by specifying the URL and the name of the package as:
```shell
pip install -e git+ssh://git@gitlab.com/slimstock-research/example_module.git#egg=example_package
```
Afterwards you can generate a "requirements.txt" file automatically that includes the required information about the installed package and the source repository.