#!/usr/bin/env python3
"""Make fws' MITM proxy honour $TMPDIR instead of a hard-coded /tmp.

@juppytt/fws 0.4.x signs a per-host TLS cert in `/tmp/fws-cert-<ts>` and
writes a copy of its CA *private key* there (mode 0644). With a read-only or
missing /tmp (enroot/Pyxis, SLURM) every intercepted HTTPS request then fails
with "502 Bad Gateway"; on a shared /tmp the CA key is briefly world-readable.
Use fs.mkdtemp(os.tmpdir()) instead (0700, honours TMPDIR). Idempotent; exits
non-zero if the upstream code changed so the build notices.
"""
import pathlib
import subprocess
import sys

root = pathlib.Path(subprocess.check_output(["npm", "root", "-g"], text=True).strip())
target = root / "@juppytt" / "fws" / "src" / "proxy" / "mitm.ts"
src = target.read_text()
old = (
    "  const tmpDir = `/tmp/fws-cert-${Date.now()}`;\n"
    "  await fs.mkdir(tmpDir, { recursive: true });\n"
)
new = (
    "  // patched (pinchbench): honour TMPDIR, private 0700 dir\n"
    "  const tmpDir = await fs.mkdtemp(path.join((await import('node:os')).tmpdir(), 'fws-cert-'));\n"
)
if new in src:
    print(f"already patched: {target}")
    sys.exit(0)
if old not in src:
    sys.exit(f"fws mitm.ts layout changed, patch not applied: {target}")
target.write_text(src.replace(old, new, 1))
print(f"patched {target}")
