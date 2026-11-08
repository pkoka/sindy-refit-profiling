<!-- rosentic:start -->
## Rosentic

Rosentic detects cross-branch compatibility conflicts on shared function
signatures, HTTP routes, and schemas.

- Before editing a shared surface, call `check_file` with the complete proposed
  file content. Validate the result and tell the user about any cross-branch risk
  before writing the change.
- Before pushing, call `check_conflicts` for the repository. Validate the result
  and tell the user which active branches need attention.
- If the Rosentic tools are unavailable, or a result is `NOT_EXECUTED` or
  `DEGRADED`, say so immediately and do not treat it as a clear result. A
  `SCOPED` result is valid, but it is not a full-repository clear - report it
  as scoped, naming what was not compared, never as a clean pass.
<!-- rosentic:end -->
