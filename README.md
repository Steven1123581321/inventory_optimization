# Use
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

In your repository, you should always create a file "requirements.txt" that contains all the required packages (dependencies) using:
```shell
pip freeze > requirements.txt
```

When setting up the repository on a new location, you can use this file to install all required packages using (first activate the virtual environment):
```shell
pip install -r requirements.txt
```

## Running the commands

To run this example package you need a configuration JSON. The defualt name is "config.json". Different commands can be implemented that can be executed by running:
```shell
python main.py example_command
```