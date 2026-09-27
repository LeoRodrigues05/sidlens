"""sidlens command line: freeze | verify | show | bundle."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime


def _cmd_freeze(args) -> int:
    from sidlens import paths
    from sidlens.provenance import freeze, manifest as manifest_mod

    snapshot_id = args.snapshot_id or datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"freeze  snapshot={snapshot_id}  dry_run={args.dry_run}")
    print(f"  source repo : {paths.UPSTREAM_REPO}")
    print(f"  source work : {paths.UPSTREAM_WORK}")
    print(f"  destination : {paths.FROZEN}\n")

    jobs = freeze.in_flight_jobs()
    if jobs:
        print("  NOTE: SLURM jobs are running against the source trees --")
        for j in jobs:
            print(f"        {j['jobid']} {j['name']} {j['state']} {j['elapsed']} {j['node']}")
        print("        recorded in the manifest as in_flight_jobs.\n")

    print("data sections:")
    sections = freeze.freeze_data(dry_run=args.dry_run, only_kind=args.kind)
    print("\nvendor:")
    vendor = freeze.freeze_vendor(dry_run=args.dry_run)

    if args.dry_run:
        print("\ndry run -- nothing written")
        return 0

    print("\nupstream state...")
    upstream = freeze.upstream_state()
    print(f"  head={upstream['head'][:12]}  dirty_paths={upstream['n_dirty_paths']}")

    man = manifest_mod.build(snapshot_id=snapshot_id, upstream=upstream,
                             vendor=vendor, data_sections=sections, in_flight=jobs)
    path = manifest_mod.save(man)
    print(f"\nmanifest -> {path}")

    if not args.no_readonly:
        n = freeze.make_read_only(paths.FROZEN)
        print(f"read-only: {n} files under {paths.FROZEN}")
        nv = freeze.make_read_only(paths.VENDOR)
        print(f"read-only: {nv} files under {paths.VENDOR}")

    print()
    print(manifest_mod.summarize(man))
    return 0


def _cmd_verify(args) -> int:
    from sidlens.provenance import verify
    report = verify.verify_snapshot(args.snapshot_id, quick=args.quick,
                                    profile=args.profile)
    print(verify.format_report(report, verbose=args.verbose))
    if not report["ok"] and args.strict:
        return 1
    return 0


def _cmd_show(args) -> int:
    from sidlens.provenance import manifest as manifest_mod
    man = manifest_mod.load(args.snapshot_id)
    print(manifest_mod.summarize(man))
    if args.sections:
        print("\nsections:")
        for dest, sec in man["data"].items():
            print(f"  {dest:42s} {sec['n_files']:5d} files "
                  f"{sec['bytes']/1e6:9.1f} MB  [{sec['kind']}]")
    return 0


def _fmt_bytes(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


def _cmd_bundle(args) -> int:
    import os
    from sidlens.provenance import bundle, profiles

    if args.action == "profiles":
        for name, prof in profiles.PROFILES.items():
            print(f"{name:8s} {prof.description}")
        return 0

    if args.action == "plan":
        p = bundle.plan(args.profile, tuple(args.extra))
        t = bundle.totals(p["files"])
        print(f"profile {p['profile']}  snapshot {p['snapshot_id']}")
        print(f"  frozen : {t['frozen_files']:5d} files  {_fmt_bytes(t['frozen_bytes'])}")
        print(f"  extra  : {t['extra_files']:5d} files  {_fmt_bytes(t['extra_bytes'])}"
              f"  ({', '.join(p['extra']) or 'none'})")
        print(f"  total  : {t['files']:5d} files  {_fmt_bytes(t['bytes'])}")
        if p["excluded"]["files"]:
            print(f"  excluded by profile ({', '.join(p['excluded']['globs'])}):")
            for rel in p["excluded"]["files"]:
                print(f"    - {rel}")
        return 0

    if args.action == "create":
        out = bundle.create(args.profile, tuple(args.extra), args.bundle_id)
        doc = bundle.load(out)
        t = doc["totals"]
        print(f"bundle {doc['bundle_id']} -> {out}")
        print(f"  {t['files']} files, {_fmt_bytes(t['bytes'])}, snapshot "
              f"{doc['snapshot_id']}, git {doc['sidlens_git']['head'][:12]} "
              f"(+{doc['sidlens_git']['dirty_paths']} dirty)")
        return 0

    if args.action == "check":
        rep = bundle.check(bundle.load(args.bundle), quick=args.quick)
        print(bundle.format_check(rep))
        return 0 if rep["ok"] else 1

    repo = args.repo or os.environ.get("SIDLENS_HF_REPO")
    if not repo:
        print("error: pass --repo <user>/<dataset> or set SIDLENS_HF_REPO", file=sys.stderr)
        return 2

    if args.action == "push":
        bundle.push(args.bundle, repo, allow_public=args.allow_public,
                    num_workers=args.workers)
        print(f"pushed {args.bundle} to {repo} (tag {args.bundle})")
        return 0

    if args.action == "pull":
        s = bundle.pull(args.bundle, repo, revision=args.revision, dry_run=args.dry_run)
        print(f"{repo}@{s['revision']}: {s['to_download']} to download "
              f"({_fmt_bytes(s['bytes_to_download'])}), {s['already_present']} already present")
        if args.dry_run:
            return 0
        print(bundle.format_check(s["check"]))
        return 0 if s["check"]["ok"] else 1
    raise AssertionError(args.action)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sidlens")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("freeze", help="build the frozen substrate (runs once)")
    f.add_argument("--snapshot-id")
    f.add_argument("--dry-run", action="store_true")
    f.add_argument("--kind", choices=("sid", "data", "ckpt", "result"))
    f.add_argument("--no-readonly", action="store_true")
    f.set_defaults(func=_cmd_freeze)

    v = sub.add_parser("verify", help="re-hash frozen/ and vendor/, report drift")
    v.add_argument("--snapshot-id")
    v.add_argument("--strict", action="store_true", help="exit 1 on any drift")
    v.add_argument("--quick", action="store_true", help="size-only, skip hashing")
    v.add_argument("--verbose", "-v", action="store_true")
    v.add_argument("--profile", default=None,
                   help="check only this transfer profile's frozen files "
                        "(core | ar | full); see `sidlens bundle profiles`")
    v.set_defaults(func=_cmd_verify)

    s = sub.add_parser("show", help="summarize a snapshot manifest")
    s.add_argument("--snapshot-id")
    s.add_argument("--sections", action="store_true")
    s.set_defaults(func=_cmd_show)

    b = sub.add_parser("bundle", help="move a profile of $SIDLENS_WORK via a private HF dataset repo")
    bsub = b.add_subparsers(dest="action", required=True)
    bsub.add_parser("profiles", help="list transfer profiles")
    for name, helptext in (("plan", "list what a profile selects; writes nothing"),
                           ("create", "hash-certify a profile and write bundles/<id>.json")):
        x = bsub.add_parser(name, help=helptext)
        x.add_argument("--profile", required=True)
        x.add_argument("--extra", action="append", default=[],
                       help="extra WORK-relative tree, e.g. derived/controlled/<exp>/<run>")
        if name == "create":
            x.add_argument("--bundle-id", default=None,
                           help="default: <profile>-<YYYYmmddTHHMMSS>")
    x = bsub.add_parser("check", help="hash-check a bundle's files in $SIDLENS_WORK")
    x.add_argument("bundle", help="bundle id or path to its manifest")
    x.add_argument("--quick", action="store_true", help="sizes only")
    x = bsub.add_parser("push", help="upload a bundle to an existing PRIVATE dataset repo")
    x.add_argument("bundle")
    x.add_argument("--repo", default=None, help="default: $SIDLENS_HF_REPO")
    x.add_argument("--workers", type=int, default=None,
                   help="upload threads (default 4; the Hub client would use all cores but two)")
    x.add_argument("--allow-public", action="store_true", help=argparse.SUPPRESS)
    x = bsub.add_parser("pull", help="download a bundle into $SIDLENS_WORK and hash-check it")
    x.add_argument("bundle")
    x.add_argument("--repo", default=None, help="default: $SIDLENS_HF_REPO")
    x.add_argument("--revision", default=None, help="default: the bundle's tag")
    x.add_argument("--dry-run", action="store_true")
    b.set_defaults(func=_cmd_bundle)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
