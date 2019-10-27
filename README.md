# API Broken Heart - FE Project

## Required services/applications:

* Python 3: [Download](https://www.python.org/Downloads/) 

## Project Setup

The following is expecting you to have python 3.x installed on your machine. I recommend
 looking that the [Hitchhikers guide to Python](http://docs.python-guide.org/en/latest/) if you 
 haven't.
 
 For windows users it's a good idea to install the Anaconda package. Anaconda is the leading open 
 data science platform powered by Python (according to their homepage) [Anaconda](https://www.continuum.io/Downloads)
 
### [OPTIONAL]Create a virtual environment for the project 

Look at the following guide for more details [guide](http://docs.python-guide.org/en/latest/dev/virtualenvs/#virtualenvironments-ref)

Run terminal/command line on project folder

* Windows

```bash
> pip install virtualenv
> virtualenv venv
> venv\Scripts\activate.bat
```

* Linux/MacOS

```bash
> pip install virtualenv
> virtualenv venv
> source venv/bin/activate
```

if you are running Anaconda you can also use conda virtual environment instead.

### Get the required packages

```bash
pip3 install -r requirements.txt
```

## Start the web server

 * Open terminal/command line

 To start the development server run:
 
```bash
> export APP_SETTINGS=local_settings.cfg
> python3 app.py
```

### Closing down.

when you are finished running the project you can:

* Close down the server by pressing <CLTR>-c  
* exit the virtual env:

```bash
> deactivate
```

