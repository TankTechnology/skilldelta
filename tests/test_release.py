"""Public packaging must never pick up locally generated records or credentials."""
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from package_release import package
from public_release import validate


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "README.md").write_text("Synthetic test repository.\n")
        (self.root / "PUBLIC_FILES.txt").write_text("README.md\nPUBLIC_FILES.txt\n")

    def test_generated_files_are_not_packaged(self):
        (self.root / "data").mkdir()
        (self.root / "data/private.json").write_text('{"synthetic": true}')
        (self.root / "unlisted.py").write_text("# unreviewed file\n")
        output = self.root / "dist/release.zip"
        package(self.root, output)
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(set(archive.namelist()), {
                "skilldelta/README.md", "skilldelta/PUBLIC_FILES.txt", "skilldelta/MANIFEST.json"})

    def test_adding_data_to_allowlist_is_rejected(self):
        with (self.root / "PUBLIC_FILES.txt").open("a") as f:
            f.write("data/private.json\n")
        with self.assertRaisesRegex(ValueError, "forbidden public path"):
            validate(self.root)

    def test_secret_report_does_not_reveal_value(self):
        synthetic_token = "sk" + "-" + "Q" * 30
        (self.root / "README.md").write_text(synthetic_token)
        with self.assertRaises(ValueError) as caught:
            validate(self.root)
        self.assertIn("possible provider token", str(caught.exception))
        self.assertNotIn(synthetic_token, str(caught.exception))

    def test_symlink_cannot_bypass_allowlist(self):
        source = self.root / "README.md"
        source.unlink()
        source.symlink_to(self.root / "PUBLIC_FILES.txt")
        with self.assertRaisesRegex(ValueError, "symlink excluded"):
            validate(self.root)


if __name__ == "__main__":
    unittest.main()
