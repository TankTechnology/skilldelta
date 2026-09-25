---
name: skilldelta-route
description: "Use the SkillDelta plugin's frozen task-conditional gate when deciding whether a supplied skill should be enabled for a user task."
---

# SkillDelta routing

The plugin performs the route automatically before the first model step of a
new user turn. Do not copy the skill text into the prompt and do not call the
route tool again unless you are explicitly auditing a decision.

The route result is an operational choice, not evidence that the task will
definitely succeed. Continue to follow the user's request and the harness
tools. If routing falls back because the embedding service or support file is
unavailable, the result includes the fallback reason in the audit log.
