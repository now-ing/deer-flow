"""Tests for deerflow.uploads.manager — shared upload management logic."""

import errno
import os
import threading
from unittest.mock import patch

import pytest

from deerflow.uploads.manager import (
    PathTraversalError,
    UnsafeUploadPathError,
    claim_unique_filename,
    cleanup_stale_upload_staging_files,
    delete_file_safe,
    list_files_in_dir,
    normalize_filename,
    validate_path_traversal,
    write_upload_file_no_symlink,
)

# ---------------------------------------------------------------------------
# normalize_filename
# ---------------------------------------------------------------------------


class TestNormalizeFilename:
    def test_safe_filename(self):
        assert normalize_filename("report.pdf") == "report.pdf"

    def test_strips_path_components(self):
        assert normalize_filename("../../etc/passwd") == "passwd"

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_filename("")

    def test_rejects_dot_dot(self):
        with pytest.raises(ValueError, match="unsafe"):
            normalize_filename("..")

    def test_strips_separators(self):
        assert normalize_filename("path/to/file.txt") == "file.txt"

    def test_dot_only(self):
        with pytest.raises(ValueError, match="unsafe"):
            normalize_filename(".")


# ---------------------------------------------------------------------------
# claim_unique_filename
# ---------------------------------------------------------------------------


class TestDeduplicateFilename:
    def test_no_collision(self):
        seen: set[str] = set()
        assert claim_unique_filename("data.txt", seen) == "data.txt"
        assert "data.txt" in seen

    def test_single_collision(self):
        seen = {"data.txt"}
        assert claim_unique_filename("data.txt", seen) == "data_1.txt"
        assert "data_1.txt" in seen

    def test_triple_collision(self):
        seen = {"data.txt", "data_1.txt", "data_2.txt"}
        assert claim_unique_filename("data.txt", seen) == "data_3.txt"
        assert "data_3.txt" in seen

    def test_mutates_seen(self):
        seen: set[str] = set()
        claim_unique_filename("a.txt", seen)
        claim_unique_filename("a.txt", seen)
        assert seen == {"a.txt", "a_1.txt"}


# ---------------------------------------------------------------------------
# validate_path_traversal
# ---------------------------------------------------------------------------


class TestValidatePathTraversal:
    def test_inside_base_ok(self, tmp_path):
        child = tmp_path / "file.txt"
        child.touch()
        validate_path_traversal(child, tmp_path)  # no exception

    def test_outside_base_raises(self, tmp_path):
        outside = tmp_path / ".." / "evil.txt"
        with pytest.raises(PathTraversalError, match="traversal"):
            validate_path_traversal(outside, tmp_path)

    def test_symlink_escape(self, tmp_path):
        target = tmp_path.parent / "secret.txt"
        target.touch()
        link = tmp_path / "escape"
        try:
            link.symlink_to(target)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                pytest.skip("symlink creation requires Developer Mode or elevated privileges on Windows")
            raise
        with pytest.raises(PathTraversalError, match="traversal"):
            validate_path_traversal(link, tmp_path)


# ---------------------------------------------------------------------------
# write_upload_file_no_symlink
# ---------------------------------------------------------------------------


class TestWriteUploadFileNoSymlink:
    def test_writes_new_file(self, tmp_path):
        dest = write_upload_file_no_symlink(tmp_path, "notes.txt", b"hello")

        assert dest == tmp_path / "notes.txt"
        assert dest.read_bytes() == b"hello"

    def test_same_name_upload_allocates_unique_name_not_overwrite(self, tmp_path):
        # Regression for #3750: a same-name upload must not silently overwrite an
        # existing file. O_EXCL atomically allocates notes_1.txt so both versions
        # are preserved on disk.
        existing = tmp_path / "notes.txt"
        existing.write_bytes(b"old contents")
        assert os.stat(existing).st_nlink == 1

        result = write_upload_file_no_symlink(tmp_path, "notes.txt", b"new contents")

        assert result == tmp_path / "notes_1.txt"
        assert existing.read_bytes() == b"old contents"
        assert result.read_bytes() == b"new contents"
        assert os.stat(existing).st_nlink == 1
        assert os.stat(result).st_nlink == 1

    def test_repeated_same_name_uploads_increment_suffix(self, tmp_path):
        # report.pdf -> report_1.pdf -> report_2.pdf, each keeps its own content.
        first = write_upload_file_no_symlink(tmp_path, "report.pdf", b"v1")
        second = write_upload_file_no_symlink(tmp_path, "report.pdf", b"v2")
        third = write_upload_file_no_symlink(tmp_path, "report.pdf", b"v3")

        assert first.name == "report.pdf"
        assert second.name == "report_1.pdf"
        assert third.name == "report_2.pdf"
        assert first.read_bytes() == b"v1"
        assert second.read_bytes() == b"v2"
        assert third.read_bytes() == b"v3"

    def test_concurrent_same_name_uploads_never_interleave_or_lose_content(self, tmp_path):
        # Concurrency regression for #3750: many same-name uploads racing for the
        # same filename must each land in their own exclusive file with their full
        # payload intact (no interleaved / truncated / mixed bytes, no shared inode).
        n_threads = 16
        payload = b"abcdef0123456789" * 256  # 4 KiB, large enough to detect tearing
        barrier = threading.Barrier(n_threads)
        results: list = [(None, None)] * n_threads

        def writer(idx: int) -> None:
            barrier.wait()  # release all threads simultaneously
            try:
                dest = write_upload_file_no_symlink(tmp_path, "race.bin", payload)
                results[idx] = (dest, None)
            except BaseException as exc:  # pragma: no cover - surface unexpected failures
                results[idx] = (None, exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        errors = [r[1] for r in results if r[1] is not None]
        assert not errors, f"some concurrent writers failed: {errors}"

        dests = [r[0] for r in results]
        names = {d.name for d in dests}
        # Every writer must have received a distinct on-disk name (no shared inode).
        assert len(names) == n_threads, f"name collision: only {len(names)} unique names for {n_threads} writers"
        # Every file must be a separate regular file with the full payload intact.
        for d in dests:
            assert d.read_bytes() == payload, f"truncated/interleaved content in {d.name}"
            assert os.stat(d).st_nlink == 1
        # No partial / garbled files left behind in the directory.
        written = [p for p in tmp_path.iterdir() if p.is_file()]
        assert len(written) == n_threads

    def test_fallback_without_no_follow_support_succeeds(self, tmp_path, monkeypatch):
        monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

        # When O_NOFOLLOW is absent (Windows), the function falls back to
        # a dual-lstat + fstat approach and succeeds.
        result = write_upload_file_no_symlink(tmp_path, "notes.txt", b"hello")
        assert result == tmp_path / "notes.txt"
        assert (tmp_path / "notes.txt").read_bytes() == b"hello"

    def test_open_uses_nonblocking_flag_when_available(self, tmp_path):
        if not hasattr(os, "O_NONBLOCK"):
            pytest.skip("O_NONBLOCK not available on this platform")
        with patch("deerflow.uploads.manager.os.open", side_effect=OSError(errno.ENXIO, "no reader")) as open_mock:
            with pytest.raises(UnsafeUploadPathError, match="Unsafe upload destination"):
                write_upload_file_no_symlink(tmp_path, "pipe.txt", b"hello")

        flags = open_mock.call_args.args[1]
        assert flags & os.O_NONBLOCK

    @pytest.mark.parametrize("open_errno", [errno.ENXIO, errno.EAGAIN])
    def test_nonblocking_special_file_open_errors_are_unsafe(self, tmp_path, open_errno):
        if not hasattr(os, "O_NONBLOCK"):
            pytest.skip("O_NONBLOCK not available on this platform")
        with patch("deerflow.uploads.manager.os.open", side_effect=OSError(open_errno, "would block")):
            with pytest.raises(UnsafeUploadPathError, match="Unsafe upload destination"):
                write_upload_file_no_symlink(tmp_path, "pipe.txt", b"hello")

        assert not (tmp_path / "pipe.txt").exists()


# ---------------------------------------------------------------------------
# list_files_in_dir
# ---------------------------------------------------------------------------


class TestListFilesInDir:
    def test_empty_dir(self, tmp_path):
        result = list_files_in_dir(tmp_path)
        assert result == {"files": [], "count": 0}

    def test_nonexistent_dir(self, tmp_path):
        result = list_files_in_dir(tmp_path / "nope")
        assert result == {"files": [], "count": 0}

    def test_multiple_files_sorted(self, tmp_path):
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "a.txt").write_text("a")
        result = list_files_in_dir(tmp_path)
        assert result["count"] == 2
        assert result["files"][0]["filename"] == "a.txt"
        assert result["files"][1]["filename"] == "b.txt"
        for f in result["files"]:
            assert set(f.keys()) == {"filename", "size", "path", "extension", "modified"}

    def test_ignores_subdirectories(self, tmp_path):
        (tmp_path / "file.txt").write_text("data")
        (tmp_path / "subdir").mkdir()
        result = list_files_in_dir(tmp_path)
        assert result["count"] == 1
        assert result["files"][0]["filename"] == "file.txt"

    def test_filters_only_upload_staging_files(self, tmp_path):
        (tmp_path / ".env").write_text("intentional dotfile")
        (tmp_path / ".upload-active.part").write_text("partial")
        (tmp_path / ".upload-note.txt").write_text("intentional upload")
        (tmp_path / "draft.part").write_text("intentional upload")
        (tmp_path / "visible.txt").write_text("visible")

        result = list_files_in_dir(tmp_path)

        assert result["count"] == 4
        assert [f["filename"] for f in result["files"]] == [".env", ".upload-note.txt", "draft.part", "visible.txt"]


# ---------------------------------------------------------------------------
# cleanup_stale_upload_staging_files
# ---------------------------------------------------------------------------


class TestCleanupStaleUploadStagingFiles:
    def test_removes_only_stale_staging_files_from_all_upload_layouts(self, tmp_path):
        legacy_uploads = tmp_path / "threads" / "thread-legacy" / "user-data" / "uploads"
        user_uploads = tmp_path / "users" / "owner-1" / "threads" / "thread-owned" / "user-data" / "uploads"
        unrelated_uploads = tmp_path / "misc" / "thread-other" / "user-data" / "uploads"
        for uploads_dir in (legacy_uploads, user_uploads, unrelated_uploads):
            uploads_dir.mkdir(parents=True)

        (legacy_uploads / ".upload-old.part").write_text("legacy partial")
        (user_uploads / ".upload-new.part").write_text("user partial")
        (unrelated_uploads / ".upload-ignore.part").write_text("outside layout")
        (legacy_uploads / ".env").write_text("intentional dotfile")
        (legacy_uploads / ".upload-note.txt").write_text("intentional upload")
        (legacy_uploads / "draft.part").write_text("intentional upload")

        removed = cleanup_stale_upload_staging_files(tmp_path)

        assert removed == 2
        assert not (legacy_uploads / ".upload-old.part").exists()
        assert not (user_uploads / ".upload-new.part").exists()
        assert (unrelated_uploads / ".upload-ignore.part").exists()
        assert (legacy_uploads / ".env").exists()
        assert (legacy_uploads / ".upload-note.txt").exists()
        assert (legacy_uploads / "draft.part").exists()


# ---------------------------------------------------------------------------
# delete_file_safe
# ---------------------------------------------------------------------------


class TestDeleteFileSafe:
    def test_delete_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("data")
        result = delete_file_safe(tmp_path, "test.txt")
        assert result["success"] is True
        assert not f.exists()

    def test_delete_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            delete_file_safe(tmp_path, "nope.txt")

    def test_delete_traversal_raises(self, tmp_path):
        with pytest.raises(PathTraversalError, match="traversal"):
            delete_file_safe(tmp_path, "../outside.txt")
