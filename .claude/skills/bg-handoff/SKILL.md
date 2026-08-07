---
name: bg-handoff
description: >-
  Hand the CURRENT conversation off to a detached background Claude agent that survives VS Code
  disconnects, SSH drops, and laptop sleep — with full context inherited, so nothing has to be
  re-derived. Use whenever the user has finished discussing a strategy in the VS Code extension
  (or any interactive session) and now wants it built autonomously: "go build this in the
  background", "keep working after I close VS Code", "detach this", "hand this off", "run this
  unattended but keep my context". Covers the verified `claude --bg --resume` mechanism, the
  mission-brief template, permission modes (and the classifier gotcha), and how to monitor,
  steer, and end the detached run. For work that instead needs SLURM job accounting or a
  supervisor restart loop, use `slurm-claude-agent`.
---

# Hand this conversation to a detached background agent

**Problem this solves.** In the VS Code extension the `claude` process runs on Jed as a child of
the extension host (`PPID = extensionHost`, sharing vscode-server's process group). When the
Remote-SSH connection drops — which happens often on this cluster — the extension host dies and
takes the agent with it. The usual workaround (spin up a fresh SLURM agent) throws away all the
context that made this conversation worth having.

**Fix.** `claude --bg --resume <session-id>` launches a *daemon-owned* agent that inherits the
entire conversation and is reparented to init. It is not attached to VS Code in any way.

## Verified mechanism (Jed, CLI 2.1.196, extension 2.1.224, 2026-08-07)

```
PID 1
└─ claude daemon run --origin transient          ← PPID 1, session leader (Ssl)
   └─ claude bg-pty-host …/pty/<id>.sock         ← real pty, so `claude attach` works
      └─ claude --session-id <new> --fork-session --resume <old> "<mission>"
```

- **Detached**: the daemon has `PPID 1`. It shares no process group with vscode-server.
  (Corroboration: other users on this login node have PPID-1 `claude` daemons up 77–88 days.)
- **Context inherited**: verified — a `--bg --resume` of a live extension session inherited
  140 message records and answered a question that was only answerable from the prior turns.
- **Fork is automatic**: `--bg --resume` silently adds `--fork-session`, giving the background
  agent a NEW session ID. Your extension session is *not* clobbered — you can keep chatting in
  VS Code while the detached agent builds.
- **Transcripts are shared**: the background agent writes to
  `~/.claude/projects/-scratch-ymeng-masif-graph/<new-id>.jsonl`, so the extension's resume
  picker can pick its finished work back up later.

## Procedure

### 1. Preconditions
- `echo "$CLAUDE_CODE_SESSION_ID"` — this is the session to fork from. (`CLAUDE_PID` is also set.)
- **Do not hand off mid-task.** The fork snapshots the transcript at launch time; work still
  in flight in this session is not carried over. Let tool calls settle first.
- If a *previous* handoff for this same work is still running, resume that one instead of
  launching a second — check `claude agents --json` first. Two agents in one working tree
  will corrupt each other's edits.

### 2. Pin the mission before launching
Nobody is watching once it detaches, so the brief has to be self-contained. If the conversation
has already pinned these, don't re-ask — just restate them for confirmation. Otherwise ask:

- **Deliverable** — what must exist when it's done.
- **Boundaries** — per `CLAUDE.md`: no GPU training or large data transfers without explicit
  human go-ahead; SLURM account `upthomae` is budget-capped; CHF-100 is the per-session ceiling
  for one unattended stretch.
- **Stop conditions** — what counts as "blocked" vs "keep trying", and when to give up.
- **Definition of done.**

### 3. Write the mission brief to a file
Long prompts on a command line are fragile and unreviewable. Write the brief, then point the
launch prompt at it:

`.claude/bg-missions/<YYYY-MM-DD>-<slug>.md`

```markdown
# Mission: <one line>

## Deliverable
<what must exist when done>

## Context
You have inherited the full conversation that produced this brief — do not re-derive it.
Read-first pointers: <docs/…, files, job ids>

## Boundaries
- No GPU job launches or large transfers without explicit human go-ahead (CLAUDE.md).
- Budget ceiling for this run: <CHF n>. Log cumulative spend in the progress log.
- <any locked D-decisions or design constraints that must not drift>

## How to work
- Invoke the `ml-research-guardrails` skill continuously: leakage checks, shuffled-label
  controls, per-complex spread, honest stop conditions. Try to break your own good news.
- Keep an append-only running log at `docs/progress/<phase>-log.md` — append the step header
  the moment you START a step, not after. It is your memory across context summarization,
  and it is how the user reviews you without attaching.
- Mirror step headers into `docs/<nn>-<phase>-user-comment.md` and re-read it at every step
  boundary; reply to any new `### 🧑 USER:` comment inline with `### 🤖 AGENT:`, act on it,
  and keep going. A comment is steering, not a stop signal. (See `slurm-claude-agent`.)
- **Commit and push often.** `/scratch` has a 30-day cleanup that deletes `.git` internals
  and untouched sources.

## Stop conditions
- Pause and post to the comment file ONLY for: a budget gate the user reserved, or a
  decision that would diverge from a locked D-decision.
- Otherwise keep going until the definition of done is met or options are exhausted.

## Definition of done
<concrete, checkable>
```

### 4. Launch

```bash
claude --bg --resume "$CLAUDE_CODE_SESSION_ID" \
  --permission-mode acceptEdits \
  --effort max \
  "Read .claude/bg-missions/<file>.md — that is your mission brief. You have inherited the
   full conversation that produced it. <one-line restatement of the task>. Begin."
```

Verified accepted: `--bg`, `--resume`, `--permission-mode acceptEdits`, `--effort max`.
`~/.claude/settings.json` already pins `model: opus` and `effortLevel: max`, so those flags are
optional. `--model` was not separately verified with `--bg`.

### 5. Report back to the user
Give them the background ID and the four commands from §"Monitoring", plus the brief's path.

## Permission modes — read this before choosing

| mode | behaviour | notes |
|---|---|---|
| `acceptEdits` | auto-accepts file edits, still prompts for other tools | **default for this skill**; verified working |
| `bypassPermissions` | no prompts at all | see gotcha below |

**Gotcha (hit and confirmed):** when *Claude* runs `claude --bg … --permission-mode
bypassPermissions`, the auto-mode classifier **blocks it** as privilege escalation. That gate is
reasonable and must not be worked around. If the user genuinely wants a fully unattended agent:

- **print the command and let the user paste it themselves** in their own terminal, or
- have them add a Bash permission rule in `.claude/settings.json` (the `update-config` skill).

Be honest about the tradeoff: an unattended `acceptEdits` agent will **stall** on the first
non-edit permission prompt with nobody to answer it. For a long autonomous build, the
user running the `bypassPermissions` command themselves is usually the right call.

## Monitoring, steering, ending

```bash
claude agents              # list all sessions (interactive + background), with state
claude agents --json       # scriptable; no TTY needed
claude logs <id>           # recent output (raw pty replay — escape codes, expect noise)
claude attach <id>         # live terminal into the running agent; Ctrl+Z detaches,
                           #   the session KEEPS RUNNING either way
claude stop <id>           # end it
```

Async steering without attaching: edit the `### 🧑 USER:` comment file (§3). That is the
lower-friction path and it leaves a written record.

To review the finished work back in VS Code, resume the background agent's **new** session ID
from the extension's resume picker.

## When NOT to use this

- **Needs SLURM accounting, a compute-node allocation, or a supervisor restart loop** →
  `slurm-claude-agent`. A login-node background agent is fine as a *conductor* (edits code,
  `sbatch`es to Kuma, polls, decides) but must not itself do heavy compute.
- **Only the terminal is dropping, not the session** → plain `tmux` + `claude --resume <id>`
  is simpler and equivalent.

## Known limits
- The daemon runs `--origin transient`. It stayed alive through testing after its sessions
  finished, but its idle-reap timeout was not characterized. Irrelevant in practice: it is
  spawned on demand and lives as long as a session does.
- Typing `/bg` *inside* the VS Code extension was not verified — the extension launches claude
  with `--output-format stream-json` (SDK mode, no pty), and backgrounding needs a pty host.
  The `--bg --resume` route in this skill sidesteps that and is verified.
