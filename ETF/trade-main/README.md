# 2026 FMA Trading Competition

## Quick Links
[Homepage](https://trade.mcfam.forum)

[Code base](https://github.com/umnfma/trade)

[Forum](https://mcfam.forum)

## Getting Started
1. [Join the forum](https://mcfam.forum/index.php?register/)
2. [Join a team](https://mcfam.forum/index.php?threads/2026-trading-competition-sign-up-thread.33/)
3. Clone the repo.
4. Add your Alpaca API keys to the env.list file found in the project root.
5. Edit the code in such a way that you will maximize returns while managing risk.
6. Run the app.

### Unix (Linux/MacOS)
```shell
git clone git@github.com/umnfma/trade && cd trade
```
### Windows
#### Command Prompt/CMD
```cmd
git clone git@github.com:umnfma/trade.git & cd trade
```
#### Powershell
```powershell
git clone git@github.com:umnfma/trade.git; cd trade
```
4. Implement your trading strategy in ```src/systrade/trading_app.py```.

## Installation
You can run this app in a python environment, but I would recommend running it
in a Docker container.  Running it in a docker container will be helpful
(maybe) for you and helpful when tracking different teams.  If you want to
learn more about Docker, feel free to google it or checkout their website
[here](https://docs.docker.com/get-started/get-docker/). In short, Docker
will manage the environment variables for you, in a neatly packaged container
on your server or wherever you choose to run it. You may notice I provided you with
a Dockerfile so that should be the only thing you need to configure the container
once you have Docker installed. (I use docker from the terminal but docker 
does also have a nice GUI called Docker Desktop).

### Run inside the terminal
You can run the app in a python environment like a python virtual environment or an environment
with a python kernel managed by anaconda. I will not be covering that here.

Once inside the python environment, to install the app run the following
command inside the root directory the project:
```shell
pip install -e .
```

Export the environment variables to the terminal environment:
```bash
export $(cat env.list)
```
NOTE: theres a dummy env.list in the repo, just add your keys to it.

Run the app:
```shell
python src/systrade/trading_app.py
```
you will begin to see the logs, but this is a very messy way to go about this.
Follow the steps in the section below for a slight improvement.

### Run With Docker (the following are all bash commands)
Go to the folders root directory (the same directory with the dockerfile, ```env.list``` and ```src/``` directory),
and run the following:
```
docker build -t systrade . && docker run --env-file env.list -d systrade
```
First it will build the image with the tag (-t) that you give it (systrade, in this case)
and then it will run the container that it built as a daemon (-d) with the environment variable list that you 
created with your Alpaca API keys (env.list).

### Docker Tips
To check if the previous command worked as intended, run:
```docker ps```
to list all currently running docker processes with their names.
The name of the container will be structured as <adjective>_<mathematician/scientist>,
like fancy_curie, relevant_turing etc. Now to see the logs of the running app, run:
```
docker logs -f <container_name> 
```
to follow the logs of the running container. I generally like to filter then since you get a lot of
debug logs, so alternatively you could run:
```
docker logs -f <container_name> | grep -iE "info|error"
```
to greatly reduce your terminal clutter. Watching the logs for a while is always fun
and will give you a good sense of just what the app is doing.

NOTE: You can view the container's logs so log at it exists. If the app runs into any problems,
it will exit and the container will be "exited" as well. You will see this if you just list the
docker containers:
```
docker container ls
```
Exited containers will be listed as to have a status of ```exited``` and probably a time for when it exited.
Remove exited containers with ``docker rm <container_name>```.

## Expert Tips & Tricks
If you are a self-proclaimed e1337 h4xor, you can also run this with your favorite process management software.
I would recommend systemd..... but unfortunately for the competition, you are expected to make it run with docker.

## Deadlines
- Registration Ends (March 14,2026 23:59 CST)
- Final code submission (March 31, 2026 23:59 CST)
- Trading begins (April 6 @ Market-Open)
- Trading ends (April 17 @ Market-Close)
