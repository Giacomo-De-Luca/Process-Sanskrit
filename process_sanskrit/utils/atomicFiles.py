"""Publish files atomically: stage a sibling, then rename it into place.

A reader of the destination sees either the previous file or the complete new
one, never a partially written one, and a reader already holding the old file
keeps its inode.  Used by the lexicon updater for both the download and the
``word_list`` rebuild; see ``documentation/database-setup.md``.
"""

from __future__ import annotations

import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _fsync_directory(path: Path) -> None:
    """Persist a directory-entry change where POSIX supports it."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _default_file_mode() -> int:
    """The mode a newly created regular file gets under the current umask."""
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


def _copy_mode_and_owner(destination: Path, staged: Path) -> None:
    """Give ``staged`` the mode (and owner, when allowed) ``destination`` has.

    ``mkstemp`` creates 0600 files.  A replacement keeps the mode the file
    already had; a brand-new file gets the ordinary umask default, so a lexicon
    downloaded as root during an image build stays readable by the unprivileged
    runtime user.  ``chown`` runs first because it clears setuid/setgid bits.
    """
    try:
        source_stat = destination.stat()
    except FileNotFoundError:
        os.chmod(staged, _default_file_mode())
        return
    if hasattr(os, "chown"):
        try:
            os.chown(staged, source_stat.st_uid, source_stat.st_gid)
        except PermissionError:
            # An unprivileged owner already creates the sibling under its own
            # uid/gid, which is the desired deployment case.
            pass
    os.chmod(staged, stat.S_IMODE(source_stat.st_mode))


@contextmanager
def scratch_file(directory: Path, *, prefix: str, suffix: str) -> Iterator[Path]:
    """Yield a fresh temporary file in ``directory`` and remove it afterwards.

    Removal runs in ``finally`` so it also covers ``KeyboardInterrupt``.  A file
    the caller has already renamed away is simply gone.
    """
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
    os.close(descriptor)
    path = Path(name)
    try:
        yield path
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def atomic_replacement(destination: Path, *, suffix: str) -> Iterator[Path]:
    """Yield a staging path that replaces ``destination`` once the block succeeds.

    The staging file is a sibling of the destination, so the final ``os.replace``
    is a same-filesystem rename.  On success the staging file takes the
    destination's mode and owner (or the umask default for a new file), is
    fsynced, renamed onto the destination, and the directory entry is fsynced.
    If the block raises, the destination is untouched and the staging file is
    removed.  Every handle on the staging file must be closed before the block
    ends, or the rename fails on Windows.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with scratch_file(
        destination.parent, prefix=f".{destination.name}.", suffix=suffix
    ) as staged:
        yield staged
        _copy_mode_and_owner(destination, staged)
        with staged.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(staged, destination)
        _fsync_directory(destination.parent)


__all__ = ["atomic_replacement", "scratch_file"]
