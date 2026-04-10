"""Mosaic viewer — PyQt6 GUI for PDS4 F ring mosaics.

Run from the f-ring/ directory:

    python mosaic_viewer/mosaic_viewer.py [options] [mosaic_name ...]
"""
from __future__ import annotations

import argparse
import os
import sys


def _normalize_argv_for_show_radii(argv: list[str]) -> list[str]:
    """Rewrite ``--show-radii -100`` to ``--show-radii=-100`` so argparse accepts it.

    Otherwise a value starting with ``-`` is parsed as another option.
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == '--show-radii':
            if i + 1 < len(argv) and not argv[i + 1].startswith('--'):
                out.append(f'--show-radii={argv[i + 1]}')
                i += 2
                continue
            out.append(arg)
            i += 1
            continue
        out.append(arg)
        i += 1
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='mosaic_viewer/mosaic_viewer.py',
        description='PyQt6 viewer for PDS4 F ring mosaics.',
    )
    p.add_argument(
        '--bundle-path', metavar='PATH',
        default=None,
        help=('Path to the PDS4 bundle root directory. '
              'Precedence: --bundle-path > $FRING_BUNDLE_PATH > '
              'pds4/bundle under the f-ring root (parent of mosaic_viewer/).'),
    )
    bkg = p.add_mutually_exclusive_group()
    bkg.add_argument(
        '--bkg-sub', dest='bkg_sub', action='store_true', default=True,
        help='Use background-subtracted mosaics (default).',
    )
    bkg.add_argument(
        '--no-bkg-sub', dest='bkg_sub', action='store_false',
        help='Use non-background-subtracted mosaics.',
    )
    p.add_argument(
        '--obsid', metavar='OBSID', nargs='+', default=None,
        help='Restrict catalog to these Cassini observation IDs.',
    )
    p.add_argument(
        '--start-obsid', metavar='ID', default='',
        help='Only include mosaics with OBSID >= START_OBSID.',
    )
    p.add_argument(
        '--end-obsid', metavar='ID', default='',
        help='Only include mosaics with OBSID <= END_OBSID.',
    )
    p.add_argument(
        '--show-radii', metavar='R1,R2,...', default='',
        help=('Comma-separated radii relative to mean core (km) to highlight '
              'as green horizontal lines. '
              'Use --show-radii=-100 or --show-radii -100 for negative values.'),
    )
    p.add_argument(
        '--verbose', action='store_true',
        help='Print extra diagnostic information.',
    )
    p.add_argument(
        'mosaic_names', metavar='mosaic_name', nargs='*',
        help=('Observation directory names to restrict the catalog to, '
              'e.g. iss_029rf_fmovie001_vims.  If none given, all mosaics '
              'in the catalog are available.'),
    )
    return p


def _resolve_bundle_path(cli_path: str | None) -> str:
    """Determine the bundle path from CLI arg, env var, or default."""
    if cli_path:
        return os.path.abspath(cli_path)
    env = os.environ.get('FRING_BUNDLE_PATH')
    if env:
        return os.path.abspath(env)
    # Default: f-ring/pds4/bundle (parent of mosaic_viewer/ directory)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(script_dir, 'pds4', 'bundle')


def main(argv=None) -> None:
    parser = _build_parser()
    raw = sys.argv[1:] if argv is None else list(argv)
    raw = _normalize_argv_for_show_radii(raw)
    args = parser.parse_args(raw)

    bundle_path = _resolve_bundle_path(args.bundle_path)
    if not os.path.isdir(bundle_path):
        parser.error(
            f'Bundle path does not exist: {bundle_path}\n'
            'Set --bundle-path or $FRING_BUNDLE_PATH.')

    if args.verbose:
        print(f'Bundle path: {bundle_path}')
        print(f'Bkg-sub: {args.bkg_sub}')

    show_radii: list[float] = []
    if args.show_radii:
        for tok in args.show_radii.split(','):
            tok = tok.strip()
            if tok:
                try:
                    show_radii.append(float(tok))
                except ValueError:
                    parser.error(f'Invalid radius value: {tok!r}')

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt

    from catalog import MosaicCatalog
    from mosaic_window import MosaicWindow

    app = QApplication(sys.argv)
    app.setApplicationName('Mosaic Viewer')
    app.setQuitOnLastWindowClosed(True)

    print('Loading mosaic catalog …', flush=True)
    try:
        catalog = MosaicCatalog(
            bundle_path=bundle_path,
            bkg_sub=args.bkg_sub,
            name_filter=args.mosaic_names if args.mosaic_names else None,
            obsid_filter=args.obsid,
            start_obsid=args.start_obsid,
            end_obsid=args.end_obsid,
        )
    except Exception as exc:
        print(f'Error loading catalog: {exc}', file=sys.stderr)
        sys.exit(1)

    records = catalog.all_records()
    if not records:
        print('No mosaics found matching the given criteria.', file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f'{len(records)} mosaics in catalog.')

    win = MosaicWindow(
        catalog=catalog,
        bundle_path=bundle_path,
        show_radii=show_radii,
    )
    win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    win.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
