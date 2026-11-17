"""PreToolUse guard for Bash and PowerShell commands (Kumar's rules, 2026-11-17, INT-98).
Denies: force or delete pushes in any form; branch deletion; gh commands that merge, approve, close, or touch settings, the API,
issues, releases, secrets or workflows; PR text with @-mentions (outside backticks). Prints a deny decision, else nothing."""
import json, re, shlex, sys
from pathlib import Path

def deny(reason):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                             "permissionDecisionReason": f"Blocked by .claude/hooks/guard_git.py: {reason}"}}))
    sys.exit(0)

def mentions(text):
    text = re.sub(r"`[^`]*`", "", text)  # code spans do not notify
    return re.findall(r"(?<![\w.`/])@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?![\w.@])", text)

cmd = (json.load(sys.stdin).get("tool_input") or {}).get("command") or ""
for seg in re.split(r"&&|\|\||;|\||\n", cmd):
    try:
        t = shlex.split(seg, posix=True)
    except ValueError:
        t = seg.split()
    t = [x for x in t if x]
    if not t:
        continue
    # skip leading env assignments and common prefixes
    while t and (re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t[0]) or t[0] in ("&", "sudo", "command")):
        t = t[1:]
    if not t:
        continue
    prog = Path(t[0]).name.lower().removesuffix(".exe")
    args = t[1:]
    if prog == "git":
        # drop global options like -C <dir>, -c k=v
        i = 0
        while i < len(args) and args[i].startswith("-"):
            i += 2 if args[i] in ("-C", "-c", "--git-dir", "--work-tree") else 1
        sub, rest = (args[i], args[i + 1:]) if i < len(args) else ("", [])
        if sub == "push":
            for a in rest:
                if a in ("-f", "--force", "--delete", "-d", "--mirror", "--prune") or a.startswith("--force") or (re.match(r"^-[a-zA-Z]+$", a) and ("f" in a or "d" in a)):
                    deny(f"force or delete push ({a})")
                if not a.startswith("-") and (a.startswith("+") or (a.startswith(":") and len(a) > 1)):
                    deny(f"force or delete refspec ({a})")
        if sub == "branch" and any(a in ("-d", "-D", "--delete") or (re.match(r"^-[a-zA-Z]+$", a) and ("d" in a.lower())) for a in rest):
            deny("branch deletion")
    if prog == "gh":
        top = args[0] if args else ""
        if top in ("repo", "api", "issue", "release", "secret", "workflow", "ruleset", "variable", "run", "gist"):
            deny(f"gh {top} is not allowed")
        if top == "pr":
            act = args[1] if len(args) > 1 else ""
            if act in ("merge", "review", "close", "reopen", "lock", "unlock", "ready"):
                deny(f"gh pr {act} is reserved for Kumar")
            if act in ("create", "edit", "comment"):
                text = " ".join(args)
                for flag in ("--body-file", "-F"):
                    if flag in args and args.index(flag) + 1 < len(args):
                        raw = args[args.index(flag) + 1]
                        f = Path(raw)
                        if not f.exists() and raw.startswith("/"):  # a Git Bash path such as /tmp/x or /c/Users/x
                            try:
                                import subprocess
                                f = Path(subprocess.run(["cygpath", "-w", raw], capture_output=True, text=True).stdout.strip())
                            except OSError:
                                pass
                        if not f.exists():
                            deny(f"cannot read the PR body file {raw} to check for @-mentions")
                        text += "\n" + f.read_text(encoding="utf-8", errors="ignore")
                m = mentions(text)
                if m:
                    deny(f"@-mention in PR text ({', '.join(sorted(set(m)))}); put names in backticks or drop the @")
