# Human Fictions in Monetary Policy

![Python application](https://github.com/mesjou/human-frictions/actions/workflows/python-app.yml/badge.svg)

Project name is a `<utility/tool/feature>` that allows `<insert_target_audience>` to do `<action/task_it_does>`.

Additional line of information text about what the project does. Your introduction should be around 2 or 3 sentences. Don't go overboard, people won't read it.

## Prerequisites

Before you begin, ensure you have met the following requirements:
<!--- These are just example requirements. Add, duplicate or remove as required --->
* You have installed the latest version of `<coding_language/dependency/requirement_1>`
* You have a `<Windows/Linux/Mac>` machine. State which OS is supported/which is not.
* You have read `<guide/link/documentation_related_to_project>`.

## Installing <project_name>

To install <project_name>, follow these steps:

Linux and macOS:
```
<install_command>
```

Windows:
```
<install_command>
```
## Using <project_name>

To use <project_name>, follow these steps:

```
<usage_example>
```

Add run commands and examples you think users will find useful. Provide an options reference for bonus points!

## Contributing to Human Frictions
To contribute to <human-frictions>, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin human-frictions/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Development

### First usage

To start, you'll need `docker` installed. When done, run this line in human-friction dir:

```bash
docker build -t human-friction:1.0 .
```

It will build docker image, used for running other commands.
You can execute scripts using Docker. Docker must be build to reflect changes.

```bash
docker run human-friction:1.0
```

### Daily development

To start development, edit files in your editor. 
You can execute scripts using Docker. 
Docker must be build to reflect changes.

```bash
docker run human-friction:1.0
```

### Packages management

If you need an additional package in docker, add it to `requirements.in` and run script `./scripts/run-refresh-requirements`.
It will refresh `requirements.txt` file, based on content of `requirements.in`. When finished, you have to rebuild your
docker image to include new packages using `docker build -t human-friction:1.0 .` command. You also have to do it if someone else updated
`requirements.txt` and you pulled it from remote repository.

### Pre-commit

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

## Testing

We are using [pytest](http://doc.pytest.org)`for tests. You can run them via:

```bash
pytest
```

## Contributors

Thanks to the following people who have contributed to this project:

* [@mesjou](https://github.com/mesjou) 💻 

You might want to consider using something like the [All Contributors](https://github.com/all-contributors/all-contributors) specification and its [emoji key](https://allcontributors.org/docs/en/emoji-key).

## Contact

If you want to contact me you can reach me at matthias.hettich@tu-berlin.de.

## License
<!--- If you're not sure which open license to use see https://choosealicense.com/--->

This project uses the following license: [MIT License](LICENSE).
