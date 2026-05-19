# Agent Skills

## Think Before Code

- Understand the task goal and the existing repo shape before editing.
- Prefer the simplest architecture that can be explained clearly.
- Use existing project conventions when files already exist.

## Test Before Code

- Define how success will be checked before making implementation changes.
- For research/planning tasks, verify claims against available sources or local artifacts.
- For coding tasks, run the narrowest useful test first when possible, then rerun after changes.
- Project workflow: write or update `.py` source files first, run tests, fix failures until tests pass, then move to the next step.
- Only summarize working `.py` code into `.ipynb` after the source modules have passed tests.
- Prefer OOP for pipeline components when it makes experiments easier to swap and compare.

## Code Simple

- Keep code easy to explain.
- Avoid unnecessary abstractions.
- Make behavior explicit and observable.
- Add comments only when they clarify non-obvious logic.
