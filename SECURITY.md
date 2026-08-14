# Security policy

## Supported versions

Security fixes are applied to the latest released version and the `main`
branch.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting feature. Do not open a public
issue containing credentials, access tokens, private dataset URLs, or an
exploitable security report.

## Credential rules

- Never commit GitHub personal access tokens, Google OAuth tokens,
  service-account JSON, or MSWEP access credentials.
- Use `gh auth login`, Colab Secrets, Application Default Credentials, or
  workload identity.
- Do not place tokens in Git remote URLs or command-line arguments.
- Revoke any token accidentally pasted into chat, logs, notebooks, or issues.
- Review notebook outputs before committing them.

The pipeline does not need a GitHub token at runtime.
