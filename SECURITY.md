# Security Policy

## Supported versions

CABench is pre-1.0; only the latest `main` is supported. Fixes are not back-ported.

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue. Use GitHub's
[private vulnerability reporting](https://github.com/micahtracht/CABench/security/advisories/new)
for this repository (Security tab → "Report a vulnerability").

Include a description, reproduction steps, and the affected commit. You can expect an
acknowledgement within a reasonable time; please allow a fix to ship before public
disclosure.

## Scope note

CABench sends prompts to third-party LLM providers and executes no untrusted code. The
most sensitive surface is credential handling: API keys are read from the environment or a
gitignored `.env` and are never written to artifacts or logs.
