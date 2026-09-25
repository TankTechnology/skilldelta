# Code-only releases

`PUBLIC_FILES.txt` explicitly lists files allowed in a public release: source,
documentation, configuration, synthetic fixtures, licenses and the teaser.
New files require an intentional list update.

```bash
make test
make package
make verify
git add -A
make check-public
```

Packaging reads only the allowlist and rejects data/output directories,
unexpected formats, symlinks and common literal credential patterns. It writes
`dist/skilldelta-public.zip`, a SHA-256 file, and `MANIFEST.json` with source
checksums. `make check-public` also checks that the Git index contains exactly
the permitted files; run it before committing or pushing.

Keep datasets, execution records, vectors, model checkpoints, ledgers and
environment files out of this list. The two tiny plugin example JSON files are
hand-written synthetic fixtures. Pattern checks supplement file review; they
cannot detect every possible secret or private datum.

Code releases and research evidence packages have different contents. Do not
publish a code release by copying a complete evidence package.
