# AGENTS.md - Your Workspace

This folder is home. Treat it that way.

## Orchestrator & Subagent Delegation Policy (Mandatory)

This workspace uses a strict two-tier model architecture configured in `~/.config/opencode/opencode.jsonc`:
- **Primary Orchestrator (`gemini-4-argon`):** High-level reasoning, mathematical design, task decomposition, review, and user communication.
- **Subagent Workers (`opencode-go/beam-free` via `subagent` tool):**
  - `explore`: Fast read-only codebase and file search.
  - `researcher`: Read-only analysis, file inspection, and background research.
  - `general`: Multi-step implementation, file edits, test execution, and synchronization.

### Rules for the Primary Agent:
1. **Delegate by default:** Whenever a task involves searching files, reading large logs/reports, writing/editing boilerplate or application code, running shell checks, or syncing folders, spawn a `subagent` (`explore`, `researcher`, or `general`) instead of executing those tool calls directly in the main session.
2. **Never set a paid `model` parameter** on `subagent` calls (unless explicitly requested by the user) so that the subagent uses its configured free model (`opencode-go/beam-free`).
3. **Strict Free-Tier Guardrail (Mandatory):** Only free models may perform heavy lifting. If a free subagent model ever hits a rate limit, usage cap, or error, **STOP immediately and ask the user if it is okay before continuing**—never silently fall back to doing heavy lifting on the primary orchestrator model.
4. **Parallelize independent subtasks:** Launch multiple `subagent` calls in parallel when tasks are independent.
5. **Report delegation transparently:** Briefly note in responses which subtasks were delegated to subagents so the user can verify two-tier execution.
6. **Zero-Bloat, Full-Quality Hand-offs:** Instruct subagents to write complete, unabridged reports, code, and outputs to disk (`reports/`, `scripts/`, etc.) so zero detail or quality is lost, and return a high-density verification digest ($\le 500$ words containing all exact metrics, tables, assertion results, and file paths) rather than pasting hundreds of raw lines into the main session context.
7. **Single-Pass Checklists:** Give `general` subagents complete end-to-end checklists (implement $\to$ test $\to$ sync `cross_check_package/` $\to$ update `SHA256SUMS.txt`) so the orchestrator reviews the verified result in a single hop.
8. **Never Trust First-Pass Subagent Output (Mandatory Adversarial Audit):** Subagent workers (`general`) frequently take shortcuts, leave dead/stubbed code (`if False:`), write vacuous tests (`if result: assert ...`), or miss edge cases in UI/state files. After EVERY implementation pass by a `general` subagent, the Primary Orchestrator MUST spawn an independent read-only `researcher` subagent to perform a brutal line-by-line code & test audit against the exact specification.
9. **Iterative Fix-and-Audit Loop Until Zero Confirmed Bugs:** Repeat the cycle (`general` fix → automated test execution → independent `researcher` line-by-line audit) as many times as needed until a final `researcher` audit pass returns **zero confirmed bugs, zero stubbed code, zero vacuous test assertions, and 100% spec compliance**. Never report a task as complete after only a single implementation pass.
10. **Pre-Handover Full Feature & Function Verification:** Before handing any work over to the user, every feature, function, edge case, UI/JS path, and end-to-end execution flow must be tested with deterministic, unconditional automated assertions (`assert ...` with zero conditional bypasses) and verified against real/live state files.

### Mandatory Zero-Trust Subagent QA & Multi-Pass Testing Loop

- After every `general` implementation pass, run an independent `researcher` adversarial audit (line-by-line code + test review vs. spec).
- Loop: `general` fix → automated tests → `researcher` audit, until the final audit reports zero confirmed bugs, zero stubbed/dead code, zero vacuous assertions, and 100% spec compliance.
- Before handover, exercise every feature/function/edge case/UI/JS path and end-to-end flow with deterministic unconditional assertions (`assert ...`, no conditional bypasses) against real/live state files.
- Never declare completion after a single implementation pass.