"""Exercise governance failure modes using disposable files, never the core."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_code_protection.py"
SPEC = importlib.util.spec_from_file_location("code_protection_guard", SCRIPT)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class CodeProtectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.file = self.root / "src/core.py"
        self.file.parent.mkdir()
        self.file.write_text("value = 1\n")
        self.entry = {
            "path": "src/core.py", "level": "L0", "reason": "scientific core",
            "sha256": hashlib.sha256(self.file.read_bytes()).hexdigest(),
        }
        self.registry = {"schema_version": 1, "files": [self.entry]}

    def check(self, registry=None, inventory=None):
        with patch.object(guard, "inventory", return_value=inventory or {"src/core.py"}):
            return guard.check(self.root, registry or self.registry)

    def test_unchanged_core_passes_without_writing(self):
        before = self.file.stat().st_mtime_ns
        self.assertEqual(self.check(), [])
        self.assertEqual(self.file.stat().st_mtime_ns, before)
        self.assertEqual(self.file.read_text(), "value = 1\n")

    def test_changed_core_fails_without_resetting_it(self):
        self.file.write_text("value = 2\n")
        errors = self.check()
        self.assertTrue(any("L0 content changed" in item for item in errors))
        self.assertEqual(self.file.read_text(), "value = 2\n")

    def test_missing_core_fails_without_recreating_it(self):
        self.file.unlink()
        self.assertTrue(any("registered file missing" in item for item in self.check()))
        self.assertFalse(self.file.exists())

    def test_unclassified_implementation_is_rejected(self):
        errors = self.check(inventory={"src/core.py", "src/new_policy.py"})
        self.assertIn("unclassified implementation file: src/new_policy.py", errors)

    def test_duplicate_and_invalid_classification_are_rejected(self):
        registry = copy.deepcopy(self.registry)
        registry["files"].append(dict(self.entry, level="unprotected"))
        errors = self.check(registry)
        self.assertTrue(any("duplicate registry path" in item for item in errors))
        self.assertTrue(any("invalid classification" in item for item in errors))

    def test_external_path_is_rejected_before_reading(self):
        registry = {"schema_version": 1, "files": [dict(self.entry, path="../outside.py")]}
        self.assertTrue(any("unsafe registry path" in item for item in self.check(registry)))

    def test_symlink_cannot_replace_registered_core(self):
        target = self.root / "alternate.py"
        target.write_bytes(self.file.read_bytes())
        self.file.unlink()
        self.file.symlink_to(target)
        self.assertTrue(any("symlink not allowed" in item for item in self.check()))

    def test_ignored_new_source_is_still_discovered(self):
        (self.file.parent / "ignored_policy.py").write_text("value = 3\n")
        with patch.object(guard.subprocess, "check_output", return_value=b"src/core.py\0"):
            self.assertEqual(guard.inventory(self.root), {"src/core.py", "src/ignored_policy.py"})

    def test_registry_self_hash_exception_is_explicit(self):
        document = self.root / "docs/code_protection.json"
        document.parent.mkdir()
        document.write_text("{}\n")
        row = {"path": "docs/code_protection.json", "level": "L0", "reason": "governance"}
        registry = {"schema_version": 1, "files": [self.entry, row]}
        self.assertIn("registry self-integrity rule changed", self.check(registry))
        row["integrity"] = "git_review_no_self_hash"
        self.assertEqual(self.check(registry), [])

    def test_controlled_file_does_not_need_a_frozen_digest(self):
        registry = {"schema_version": 1, "files": [dict(self.entry, level="L1")]}
        self.file.write_text("value = 4\n")
        self.assertEqual(self.check(registry), [])


if __name__ == "__main__":
    unittest.main()
