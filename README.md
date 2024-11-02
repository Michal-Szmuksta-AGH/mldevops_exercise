# Development Operations for a Machine Learning Project

*(Heavily based on the [leggedrobotics/plr-exercise](https://github.com/leggedrobotics/plr-exercise) repository)*

> [!NOTE]
> In this laboratory session, we will refine your previous ML project. For reference, consider the [previous mini-projects](https://home.agh.edu.pl/~mdig/dokuwiki/doku.php?id=teaching:data_science:ml_en:topics:nn_intro).

---

## Prerequisites

- A [github.com](https://github.com/) account
- A computer with GPU or a Google Account for [Colaboratory](https://colab.research.google.com)
- An existing machine learning project

## 1. Dependency management

Before proceeding, you must define the development environment for your Python-based project. There are two main approaches:

1. **Containerization** - Create a Dockerfile to define an image with all dependencies.
2. **Virtual Environment** - Set up a virtual environment to isolate Python packages from the OS and other projects.

### Containerization

If you have access to a local machine with admin privileges, containerization (e.g., with [Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) or [Podman](https://podman.io/docs/installation)) is recommended.

### Virtual environment

If you are a standard user on the local machine, please proceed with a virtual environment:
```bash
# Create a folder for virtual environments
mkdir ~/venv
# Create the virtual environment
python3 -m venv ~/venv/mldevops
# Test the virtual environment
source ~/venv/mldevops/bin/activate
which python

# Create an alias for easier sourcing (edit the ~/.bashrc file)
nano ~/.bashrc
# Add the following line at the end of the file and save it
alias venv_plr="source ~/venv/plr/bin/activate"
```

### Remote development approach

In cases where the local machine is not prepared for project development and installing new software is not possible, **it can be used as an SSH terminal to access a remote development environment**.

Alternatively, you can use a web browser to access remote IDEs such as *Jupyter Lab* instance or [Google Colaboratory](https://colab.research.google.com). Note, however, that it is generally not recommended to work exclusively in Jupyter Notebooks, as various issues may arise.

The following exercises can be completed using only Google Colaboratory, though this should be viewed as a temporary solution rather than a *best practice*.

You will need to interact with `git` using commands like the following:
```bash
!git config --global user.email "student@student.agh.edu.pl"
!git config --global user.name "student"
!git clone https://github.com/vision-agh/mldevops_exercise.git
!cd mldevops_exercise && git status
!cd mldevops_exercise && git add filename.txt
!cd mldevops_exercise && git commit -m "message"
!cd mldevops_exercise && git push
```

## 2. Working with the Git repository

To begin, use Git for version control on your project:
1. Create a fork of this project on GitHub.
2. Clone the repository via SSH:
```bash
mkdir ~/ws
cd ~/ws
git clone git@github.com:vision-agh/mldevops_exercise.git
```
> (Replace the `vision-agh` with your `github_username`).

> [!IMPORTANT]
> Cloning via `https` allows only for pulling the code. For both pulling and pushing, use the `ssh` protocol. Instructions for setting up SSH keys and adding them to your GitHub account are available [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

1. Copy your previous ML project files (`.py` and `.ipynb`) the the root of the cloned repository.
```bash
cd ~/ws/mldevops_exercise
# Copy here your project
```

1. Commit and push your changes to the origin (i.e. GitHub) repository.
```bash
# We use the dot to add all files. Note, that it is not a typical practice.
git add .
git commit -m "initial project commit"
git push
```

1. Enable the `Issues` feature in your GitHub repository:
   - Open `https://github.com/GITHUB_USERNAME/mldevops_exercise`.
   - Go to `Settings` on the right.
   - Enable the `Issues` feature (the other features can be disabled).

2. Secure your default branch (`main`/`master`) from accidental commits:
   - Open `https://github.com/GITHUB_USERNAME/mldevops_exercise`.
   - Go to `Settings` on the right.
   - Select `Branches` on the left menu.
   - Click on the `Add branch ruleset` button.
   - Name the ruleset "default," set the enforcement status to "enabled," and configure the Targets with the `Add target` dropdown by selecting the "Include default branch" option.
   - For the options, enable the following: `Restrict deletions`, `Require a pull request before merging`, and `Block force pushes`.
   - Finish by clicking the `Create` button.

After completing the setup, you are ready to proceed with the exercises.

## 3. Project Workflow

1. For each task, create a branch named `feature/task_X`.
2. Commit all changes (**and only those changes**) related to the specific task to its branch, then push them to GitHub.
3. To complete a task, create a pull request (PR) from `feature/task_X` to `main`. Set the PR title to the task description (see below).
4. **Do not delete the branches** after merging the PR.

Tasks:
 * **Task 1**: Improve formatting using `black`.
 * **Task 2**: Set up Pre-commit to automate formatting.
 * **Task 3**: Create a Python package for your project.
 * **Task 4**: Add an online logging framework.
 * **Task 5**: Use Optuna to perform hyperparameter search.
 * **Task 6**: Add docstrings and type annotations to every Python file.
---

## Task 1 - *Any Color You Like*

Your first task is to install `black` and format the code. For more information, visit: [https://github.com/psf/black](https://github.com/psf/black).

```bash
pip3 install black
black --line-length 120 ~/ws/mldevops_exercise
```

Now everything should look well-formatted.

---
## Task 2 - Pre-commit

While it's possible to run `black` manually, relying on memory before every commit can be unreliable. Fortunately, automation with `Pre-commit` makes this easier.

Begin by following the official [quick-setup guide](https://pre-commit.com/#introduction).

```bash
pip3 install pre-commit
pre-commit --version
```

In the repository, there is an already-prepared `.pre-commit-config.yaml` file. Inspect it—this file contains the necessary configuration for `Pre-commit`.

Next, register the `pre-commit` command as a Git hook, which will automatically run each time you use `git commit`:

```bash
# Register the hook
pre-commit install
# Run the pre-commit on all files
pre-commit run --all-files
```

You may notice many changes, particularly regarding whitespace in your code. It should now appear much cleaner (at least from `git`'s perspective).

To further extend automation, add tools like `black` (for formatting), `codespell` (to fix typos), and `pyupgrade` (to update syntax to Python 3.10).

```yaml
  # Black formatter
  - repo: https://github.com/psf/black
    rev: 24.4.0
    hooks:
      - id: black
        args: ["--line-length=120"]

  # Codespell - Fix common misspellings in text files.
  - repo: https://github.com/codespell-project/codespell
    rev: v2.2.6
    hooks:
      - id: codespell
        args: [--write-changes]

  # Pyupgrade - automatically upgrade syntax for newer versions of the language.
  - repo: https://github.com/asottile/pyupgrade
    rev: v3.15.2
    hooks:
      - id: pyupgrade
        args: [--py310-plus]
```

---
## Task 3 - Poetry dependency & build manager

There are two main systems for dependency management and package building in Python: `setuptools` and `poetry`. As a rule of thumb, if your project requires complex builds (e.g., with Python bindings for dynamic C/C++ libraries), `setuptools` is a suitable choice. However, for many modern Python projects, `poetry` offers simpler configuration, making both dependency management (which can simplify Dockerfiles) and package building easier.

Explore the `poetry` [Introduction](https://python-poetry.org/docs/) and [Basic Usage](https://python-poetry.org/docs/basic-usage/) to set up dependency management and enable package building.

```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install poetry
```

Remember to update the `.gitignore` file to exclude any necessary files from Git tracking.

---
## Task 4 - Logging experiment with Weights & Biases (wandb)

Add the *Weights & Biases* (`wandb`) logger to track and visualize your experiments.

```bash
pip3 install wandb
```

1. Follow the [official wandb guide](https://docs.wandb.ai/quickstart).
2. Log `training_loss`, `validation_loss`, and *your code* as an artifact.
3. Capture a screenshot of a run showing the loss curve and the uploaded artifact.
4. Commit this screenshot to the repository.

---
## Task 5 - Hyperparameter Optimization

Use `optuna` to find the best hyperparameters (e.g., `learning rate` or `epochs`).

```bash
pip3 install optuna
```

Refer to the [official examples](https://optuna.org/#code_examples) and conduct a hyperparameter search. Here is a small example:

```python3
import optuna

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)

study.best_params  # E.g. {'x': 2.002108042}
```

---
## Task 6: Docstrings & Typing

1. Add docstrings to all public or non-trivial classes and functions. Refer to PEP 8 (Style Guide for Python Code) for guidance: https://peps.python.org/pep-0257/
2. Add type annotations to every function and method in your project. This article provides a good introduction: https://realpython.com/python-type-checking/#hello-types

---
## Additional Topics

- Linting
- Automatic testing
- Automation (GitHub Actions)
- CI/CD (Continuous Integration/Continuous Delivery)

---
## Sources:

- [github.com/leggedrobotics/plr-exercise](https://github.com/leggedrobotics/plr-exercise) by @JonasFrey96
