# Documentation policy

Status: mandatory repository policy
Updated: 2026-07-12

Documentation is a required part of implementation. A change is not ready for
handoff when its commands, defaults, interfaces, status, evidence, limitations,
or rollback instructions are knowingly stale.

## Authoritative sources

Use the narrowest authoritative source and make overview documents derive from
it rather than inventing a second status:

| Subject | Authoritative source |
|---|---|
| Machine-readable release state | `release/readiness/0.1.0.dev0.json` and `release/manifests/0.1.0.dev0.json` |
| Product label and default | `docs/release/v1_capability_matrix.md` |
| Release gates and critical path | `docs/release/v1_status.md` |
| Claim-to-evidence mapping | `docs/release/v1_claims_inventory.md` and `docs/release/v1_evidence_index.md` |
| Operator workflow maturity | `docs/golden_paths.md` and `docs/release/v1_operator_guide.md` |
| Current moonshot measurements | `docs/moonshot_status.md` and its referenced result manifest |
| Public orientation | `README.md`; it must summarize, not supersede, the sources above |
| Interface catalogue | `docs/entry_points.md`; historical or research paths must be clearly labeled |

Conceptual and research documents explain mechanisms. They are not evidence of
availability, correctness, performance, or promotion.

## Status vocabulary

- **Candidate GA core**: intended core, but not GA until every release gate and
  signed release decision passes.
- **Preview**: measured and user-accessible only through an explicit opt-in; not
  the shipped default unless the capability matrix says otherwise.
- **Experimental**: incomplete evidence and an explicit experimental boundary.
- **Labs** or **research-supported**: research code or evidence outside the v1
  support contract and first-run path.
- **Rejected/frozen**: may be retained as negative evidence but cannot be
  activated by ordinary configuration.
- **In development / unpromoted**: implementation is incomplete or its required
  evidence has not passed. It must remain default-off and must not be presented
  as Preview, supported, or production-ready.

Only the capability matrix can assign a product label or change a default.
Status documents may describe implementation progress without promoting it.

## Same-change update matrix

| Change | Required documentation and evidence review |
|---|---|
| CLI, Python API, protocol, or configuration | README or entry-point examples when public; golden path/operator guide; compatibility and rollback notes |
| Runtime behavior, batching, routing, cache, kernel, or performance | Current status document; benchmark artifact and manifest when making a measured claim; capability/claims documents if label or default could change |
| Default, promotion, deprecation, or rejection | Capability matrix, claims inventory, status, golden paths, operator guide, changelog, and machine-readable release state as applicable |
| Model, dependency, platform, build, repository contents, or artifact location | README support boundary, manifests/locks, operator guide, status, portable path controls, licensing boundary, and reproducible verification commands |
| Privacy, security, persistence, deletion, or networking | SECURITY/support/operator documentation, threat or data-lifecycle material, claims inventory, and negative/failure evidence |
| Research algorithm or prototype | Its technical document plus an explicit Labs, Experimental, or unpromoted boundary; do not add release claims without promotion evidence |
| Documentation-only correction | All documents that repeat the corrected fact; do not leave conflicting copies |
| Internal refactor with no observable change | No forced prose edit, but record the no-documentation-impact rationale and verify governed links |

Every measured statement must name or link the eligible artifact, identify the
baseline, workload, host/runtime scope, correctness condition, and relevant
limitation. Avoid floating test counts in overview pages unless tied to a dated,
immutable evidence snapshot; otherwise link to the status or evidence index.
Token-trajectory hashes may be collected by an explicit evidence harness but
must remain disabled in ordinary serving so verification telemetry does not add
production compute, retained state, or content-derived identifiers.

## Workflow

1. At task start, identify which authoritative documents the work can affect.
2. During implementation, keep new capabilities labeled unpromoted and
   default-off until their gates pass.
3. When evidence is generated, validate it, add it to the applicable evidence
   manifest, and update the status and claim boundary in the same change.
4. Before handoff, search for repeated stale terms, commands, counts, labels,
   and defaults. Reconcile conflicts at their authoritative source.
5. In the handoff, list documentation changed, evidence added, verification
   performed, and any intentionally deferred promotion requirement.

## Verification

Run at least. `verify_docs.py` checks every tracked Markdown file and resolves
the repository root from its own location, so the same command must work in a
fresh clone rather than only on the reference host:

```bash
python3 scripts/verify_docs.py
python3 scripts/verify_release_links.py
python3 scripts/verify_portable_paths.py
git diff --check
```

When release artifacts or claims change, also run:

```bash
python3 scripts/verify_release_manifest.py release/manifests/0.1.0.dev0.json
```

Run the focused and full relevant code suites separately. Passing tests prove
the behavior they cover; they do not, by themselves, prove a performance claim
or authorize a status/default change.
