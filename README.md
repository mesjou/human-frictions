# Human Fictions in Monetary Policy

![Python application](https://github.com/mesjou/human-frictions/actions/workflows/python-app.yml/badge.svg)

The project enables economists to simulate how human frictions influence responses to monetary policy.
The Gym-style environment reflects classic macro-economic model with human agents representing households.
The agents can learn to respond optimally with simple heuristics or deep reinforcement learning (e.g. with rllib).

## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have a `Linux/Mac` machine.
* You have `docker` installed.

## Development

#### First usage

To start, run this line in human-friction dir:

```bash
docker build -t human-friction:1.0 .
```

It will build docker image, used for running other commands.
You can execute scripts using Docker. Docker must be build to reflect changes.

```bash
docker run human-friction:1.0
```

#### Daily development

To start development, edit files in your editor.
You can execute scripts using Docker.
Docker must be build to reflect changes.

```bash
docker run human-friction:1.0
```

### Alternatively: Virtualenv instead of Docker

```bash
python3 -m venv venv
. venv/bin/activate
pip3 install --upgrade pip setuptools wheel
pip3 install -e .
pip3 install -r requirements.txt
```

Then you can run a training script from command line, e.g.:
```bash
python3 train.py
```

#### Pre-commit

Make sure that code style follows our guidlines by installing pre-commit:
Using pip:
```bash
pip install pre-commit
```
Using homebrew:
```bash
brew install pre-commit
```
Then run
```bash
pre-commit install
```
to set up the git hook scripts.

#### Packages management

If you need an additional package in docker, add it to `requirements.in`
and run script `./scripts/run-refresh-requirements`.
It will refresh `requirements.txt` file, based on content of `requirements.in`.
When finished, you have to rebuild your docker image to include new packages
using `docker build -t human-friction:1.0 .` command.
You also have to do it if someone else updated
`requirements.txt` and you pulled it from remote repository.

## Using human-friction

To use human-friction, train and evaluate the agents, an example is provided in ```train.py``` and ```evaluate.py```.

## Contributing to Human Frictions
To contribute to human-frictions, follow these steps:

1. Clone this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).



## Testing

We are using [pytest](http://doc.pytest.org)`for tests. You can run them via:

```bash
pytest
```

## Contributors

Thanks to the following people who have contributed to this project:

* [@mesjou](https://github.com/mesjou) 💻
* [@annaalm](https://github.com/annaalm) 💻

## Contact

If you want to contact me you can reach me at matthias.hettich@tu-berlin.de.

## License

This project uses the following license: [MIT License](LICENSE).
