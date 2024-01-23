# Developer Guidelines

## Pull Requests
Before contributing to the `main` branch, make sure the following requirements are met:
- [ ] All existing tests have passed
- [ ] No local merge conflicts
- [ ] Code is commented
- [ ] Added tests for new features

Once you finish making changes on your branch, create a pull request.
To create the pull request, follow the steps:

1. Pull from `main` branch
```sh
git pull # Get most up-to-update version
```
2. Merge with main and check for conflicts
```sh
git merge main # merge with main locally on your branch to check for conflicts
```
3. Run tests
```sh
pytest # standard testing
pytest --disable-warnings # tests without displaying warnings
pytest -rP # tests while displaying print statements
```

4. Push changes

```sh
git push
```

5. Create pull request on GitHub with descriptions for changes.
 - Fill out pull request template
 - Assign reviewers (repo admin(s) by default)

## Pull Request Review
When a pull request is submitted, reviews should be done in a timely manner (it will be time-consuming to remember what the code does if the code takes a month to be reviewed)

When reviewing a pull request, approve when:
 - Requirements above are met
   - Make sure code is commented cleanly 
 - GitHub actions tests pass
 - The code runs locally on your computer (run cases to try and break the new code)

When rejecting a pull request, write a precise description of what needs to improved/fixed in the comment section.

## Tests
For test driven development, create tests before code implementation.
Templates from `tests/template.py` can be used. Create tests in `tests` directory and write test functions with `test_` prefix in test files.