# Adding this AI code to a teammate's collaborative GitHub repository

Assume the shared repository already exists.

## 1. Accept the collaborator invitation

Open the GitHub invitation and accept it.

## 2. Clone the repository

```bash
git clone https://github.com/OWNER/REPOSITORY.git
cd REPOSITORY
```

## 3. Create the integration branch if your team does not already have it

```bash
git checkout -b develop
git push -u origin develop
```

If `develop` already exists:

```bash
git fetch origin
git checkout develop
git pull origin develop
```

## 4. Create your AI feature branch

```bash
git checkout -b feature/ai-context-service
```

## 5. Copy this project into the shared repository

A clean team layout is:

```text
REPOSITORY/
├── ml/
│   └── teammate ML code
├── ai/
│   └── this smart-belt-ai code
├── hardware/
│   └── ESP32 / sensor code
└── README.md
```

So copy all files from this package into:

```text
REPOSITORY/ai/
```

## 6. Commit your AI code

```bash
git status
git add ai/
git commit -m "Add AI context-aware risk assessment service"
git push -u origin feature/ai-context-service
```

## 7. Open a Pull Request

On GitHub:

```text
feature/ai-context-service -> develop
```

Ask your teammate to review it.

## 8. Merge to develop

After testing both ML and AI together, merge the PR.

## 9. Create smaller branches for future work

From `develop`:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/dashboard
```

Work, then:

```bash
git add .
git commit -m "Add smart belt monitoring dashboard"
git push -u origin feature/dashboard
```

Open:

```text
feature/dashboard -> develop
```

Repeat for:

```text
feature/api
feature/risk-engine
feature/alert-service
feature/ml-integration
feature/tests
```

## 10. Final stable merge

Once the complete project works:

```text
develop -> main
```

Use a Pull Request instead of force-pushing to main.
