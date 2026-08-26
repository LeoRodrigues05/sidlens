"""sidlens command line: freeze | verify | registry | run."""

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
    report = verify.verify_snapshot(args.snapshot_id, quick=args.quick)
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
    v.set_defaults(func=_cmd_verify)

    s = sub.add_parser("show", help="summarize a snapshot manifest")
    s.add_argument("--snapshot-id")
    s.add_argument("--sections", action="store_true")
    s.set_defaults(func=_cmd_show)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
