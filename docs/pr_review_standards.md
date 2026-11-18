# Pull Request & Review Standards

## Branch rules
- Never push directly to `main`. Work only on your own branch.
- Every member makes at least one commit per day on their branch.
- Pull `main` into your branch regularly: `git pull origin main`.
- Keep the agreed folder structure. Put code only in your assigned files.

## Opening a Pull Request
1. Push your branch: `git push origin <your-branch>`
2. On GitHub, click **Compare & pull request** (base: `main`).
3. Fill in the PR template completely.
4. Make sure `python -m pytest` passes before requesting review.

## Review standards (reviewer checks)
| Check | Pass when |
|---|---|
| Scope | Only the author's assigned files are changed |
| Structure | No folders renamed, moved, or added without agreement |
| Tests | `python -m pytest` passes with no failures |
| Pipeline | `python -m src.orchestrator` still runs end to end |
| Contracts | Function names and return types match `docs/role_specifications.md` |
| Safety | No API keys, `.env`, or large data files committed |
| Readability | Clear names and short docstrings on public functions |

## Merging
- At least one approval from the Team Lead is required.
- Merge only when all checks above pass.

## How the Team Lead turns on branch protection (repo admin only)
1. GitHub repo -> **Settings** -> **Branches** -> **Add branch protection rule**.
2. Branch name pattern: `main`.
3. Tick **Require a pull request before merging**.
4. Tick **Require approvals** (set to 1).
5. Save the rule.

The PR template is `.github/pull_request_template.md` (GitHub fills it in automatically for every new Pull Request).
