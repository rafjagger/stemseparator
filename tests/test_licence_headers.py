# SPDX-FileCopyrightText: 2026 Raphael Eismann
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPDX = "SPDX-License-Identifier: GPL-3.0-or-later"


def python_files():
    for folder in ("src", "tests"):
        yield from (ROOT / folder).rglob("*.py")
    modal_app = ROOT / "modal_app.py"
    if modal_app.exists():
        yield modal_app


class LicenceHeaderTest(unittest.TestCase):
    def test_every_python_file_has_spdx_header(self):
        missing = []
        for path in python_files():
            head = path.read_text(encoding="utf-8").splitlines()[:3]
            if not any(SPDX in line for line in head):
                missing.append(str(path.relative_to(ROOT)))
        self.assertEqual(missing, [], "files without SPDX header")


if __name__ == "__main__":
    unittest.main()
