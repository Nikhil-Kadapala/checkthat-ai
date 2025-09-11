# Contributing to CheckThat AI Python SDK

Thank you for your interest in contributing to the CheckThat AI Python SDK! We welcome contributions from the community to help improve this unified LLM API with fact-checking capabilities. This guide outlines our contribution process and best practices.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct.html). By participating, you are expected to uphold this code. Please report unacceptable behavior to [kadapalanikhil@gmail.com](mailto:kadapalanikhil@gmail.com).

## How to Contribute

### Reporting Bugs

- Check if the issue already exists in the [issue tracker](https://github.com/nikhil-kadapala/checkthat-ai/issues).
- If not, create a new issue with:
  - A clear, descriptive title.
  - Steps to reproduce the bug.
  - Expected vs. actual behavior.
  - Your environment (Python version, OS, SDK version).
  - Any relevant logs or screenshots.

### Suggesting Enhancements

- Open an issue with the "enhancement" label.
- Describe the current behavior and why the enhancement would be useful.
- Provide implementation ideas if possible.

### Pull Requests

1. Fork the repository and create a new branch from `main` (e.g., `feature/new-endpoint` or `bugfix/issue-123`).
2. Make your changes, following the best practices below.
3. Ensure your code passes all tests and linters.
4. Update documentation if necessary.
5. Commit your changes with clear, descriptive messages.
6. Push to your fork and submit a pull request to the `main` branch.
7. Reference any related issues in the PR description.

We may ask for changes before merging. All contributions are licensed under the project's MIT license.

## Development Best Practices

Follow these guidelines to ensure high-quality contributions. These are based on our [Python SDK Development Best Practices](#python-sdk-development-best-practices).

### Code Organization and Structure

- Organize code into logical modules (e.g., authentication, API endpoints, utilities).
- Use consistent naming: classes in CapWords, functions/variables in lowercase_with_underscores.
- Maintain the project structure: source in `checkthat_ai/`, tests in `tests/`, docs in `docs/`.

### Coding Style

- Adhere to PEP 8: 4 spaces indentation, max 88-character lines.
- Use type hints for all functions and methods.
- Write docstrings for public elements following PEP 257.
- Group imports: standard library first, then third-party, then local.

### Documentation

- Update inline docstrings for any changed code.
- If adding new features, update README.md and any relevant docs.
- Use Sphinx-style for API documentation.

### Error Handling

- Raise specific exceptions (e.g., ValueError for invalid inputs).
- Use try-except blocks appropriately; avoid broad except clauses.
- Log errors with context using the logging module.

### Testing

- Write unit tests for new features using pytest.
- Aim for >80% coverage; run `pytest --cov`.
- Test edge cases and error conditions.
- Use mocks for external API calls.

### Packaging and Dependencies

- Update `pyproject.toml` for new dependencies.
- Follow Semantic Versioning for releases.

### Security

- Validate all inputs.
- Avoid storing or logging sensitive information.
- Use secure practices for API interactions (e.g., HTTPS).

## Setting Up Development Environment

1. Clone the repository: `git clone https://github.com/nikhil-kadapala/checkthat-ai.git`
2. Install dependencies: `pip install -e ".[dev]"`
3. Run linters: `black . && isort . && mypy . && flake8 .`
4. Run tests: `pytest`

## Questions?

If you have questions, open an issue or contact the maintainers.

Thank you for contributing to CheckThat AI!
