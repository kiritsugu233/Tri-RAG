import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from tri_rag_harness.beir_dataset import (
    BEIRDatasetError,
    load_beir_dataset_config,
    prepare_beir_dataset,
)


ROOT = Path(__file__).resolve().parents[1]


class BEIRDatasetTests(unittest.TestCase):
    @staticmethod
    def _write_archive(
        path: Path,
        *,
        development_rows=None,
        test_rows=None,
    ) -> None:
        corpus = [
            {"_id": f"d{index}", "title": f"Title {index}", "text": f"Body {index}"}
            for index in range(4)
        ]
        queries = [
            {"_id": f"q{index}", "text": f"Claim {index}"}
            for index in range(6)
        ]
        if development_rows is None:
            development_rows = [
                ("q0", "d0", 1),
                ("q1", "d1", 2),
                ("q2", "d2", 1),
                ("q3", "d3", 1),
            ]
        if test_rows is None:
            test_rows = [("q4", "d0", 1), ("q5", "d1", 1)]

        def jsonl(rows):
            return "".join(json.dumps(row) + "\n" for row in rows)

        def qrels(rows):
            return "query-id\tcorpus-id\tscore\n" + "".join(
                f"{query_id}\t{doc_id}\t{score}\n"
                for query_id, doc_id, score in rows
            )

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("fixture/corpus.jsonl", jsonl(corpus))
            archive.writestr("fixture/queries.jsonl", jsonl(queries))
            archive.writestr("fixture/qrels/train.tsv", qrels(development_rows))
            archive.writestr("fixture/qrels/test.tsv", qrels(test_rows))

    @staticmethod
    def _write_config(path: Path, archive: Path, *, archive_md5=None) -> None:
        if archive_md5 is None:
            archive_md5 = hashlib.md5(archive.read_bytes()).hexdigest()
        raw = {
            "schema_version": 1,
            "adapter": "beir_zip_v1",
            "dataset_name": "Fixture",
            "dataset_version": "fixture-v1",
            "id_namespace": "fixture",
            "minimum_relevance": 1,
            "source": {
                "url": "https://example.invalid/fixture.zip",
                "archive_md5": archive_md5,
                "archive_root": "fixture",
                "corpus_member": "corpus.jsonl",
                "queries_member": "queries.jsonl",
                "qrels_members": {
                    "development": "qrels/train.tsv",
                    "test": "qrels/test.tsv",
                },
                "homepage": "https://example.invalid/fixture",
                "licenses": [
                    {
                        "component": "fixture",
                        "identifier": "CC0-1.0",
                        "url": "https://example.invalid/license",
                    }
                ],
            },
            "splits": {
                "seed": 17,
                "development_qrels": "development",
                "test_qrels": "test",
                "tune_fraction": 0.5,
            },
        }
        path.write_text(json.dumps(raw), encoding="utf-8")

    @staticmethod
    def _jsonl(path: Path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_prepare_is_reproducible_external_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            archive = directory / "fixture.zip"
            config_path = directory / "config.json"
            self._write_archive(archive)
            self._write_config(config_path, archive)
            config = load_beir_dataset_config(config_path)

            first = prepare_beir_dataset(config, archive, directory / "first")
            second = prepare_beir_dataset(config, archive, directory / "second")
            self.assertEqual(set(first), set(second))
            for name in first:
                self.assertEqual(first[name].read_bytes(), second[name].read_bytes())

            manifest = json.loads(first["dataset_manifest.json"].read_text())
            queries = self._jsonl(first["queries.jsonl"])
            corpus = self._jsonl(first["corpus.jsonl"])
            qrels = self._jsonl(first["qrels.jsonl"])
            splits = json.loads(first["splits.json"].read_text())
            self.assertEqual(manifest["counts"], {"corpus": 4, "qrels": 6, "queries": 6})
            self.assertEqual(
                {name: len(ids) for name, ids in splits.items()},
                {"query_cert": 2, "query_test": 2, "query_tune": 2},
            )
            split_sets = [set(ids) for ids in splits.values()]
            self.assertFalse(
                any(
                    left.intersection(right)
                    for index, left in enumerate(split_sets)
                    for right in split_sets[index + 1 :]
                )
            )
            corpus_ids = {row["doc_id"] for row in corpus}
            query_ids = {row["query_id"] for row in queries}
            self.assertFalse(corpus_ids.intersection(query_ids))
            self.assertTrue(all(value.startswith("fixture:doc:") for value in corpus_ids))
            self.assertTrue(all(value.startswith("fixture:query:") for value in query_ids))
            self.assertEqual({row["query_id"] for row in qrels}, query_ids)
            self.assertTrue(all(row["doc_id"] in corpus_ids for row in qrels))
            self.assertTrue(manifest["ids"]["queries_are_external"])
            self.assertTrue(manifest["ids"]["splits_are_disjoint"])
            self.assertFalse(manifest["split_rule"]["uses_qrel_labels"])
            self.assertEqual(
                manifest["source"]["archive"]["md5"],
                hashlib.md5(archive.read_bytes()).hexdigest(),
            )

    def test_archive_checksum_mismatch_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            archive = directory / "fixture.zip"
            config_path = directory / "config.json"
            output = directory / "output"
            self._write_archive(archive)
            self._write_config(config_path, archive, archive_md5="0" * 32)
            config = load_beir_dataset_config(config_path)
            with self.assertRaisesRegex(BEIRDatasetError, "archive MD5 mismatch"):
                prepare_beir_dataset(config, archive, output)
            self.assertFalse(output.exists())

    def test_missing_qrel_document_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            archive = directory / "fixture.zip"
            config_path = directory / "config.json"
            output = directory / "output"
            self._write_archive(
                archive,
                development_rows=[
                    ("q0", "missing", 1),
                    ("q1", "d1", 1),
                    ("q2", "d2", 1),
                    ("q3", "d3", 1),
                ],
            )
            self._write_config(config_path, archive)
            config = load_beir_dataset_config(config_path)
            with self.assertRaisesRegex(BEIRDatasetError, "missing_docs=.*missing"):
                prepare_beir_dataset(config, archive, output)
            self.assertFalse(output.exists())

    def test_development_test_query_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            archive = directory / "fixture.zip"
            config_path = directory / "config.json"
            output = directory / "output"
            self._write_archive(archive, test_rows=[("q0", "d0", 1), ("q5", "d1", 1)])
            self._write_config(config_path, archive)
            config = load_beir_dataset_config(config_path)
            with self.assertRaisesRegex(BEIRDatasetError, "query IDs overlap"):
                prepare_beir_dataset(config, archive, output)
            self.assertFalse(output.exists())

    def test_checked_in_scifact_config_is_pinned(self):
        config = load_beir_dataset_config(
            ROOT / "configs" / "real_scifact_dataset.json"
        )
        self.assertEqual(config.dataset_name, "BEIR SciFact")
        self.assertEqual(
            config.source.archive_md5, "5f7d1de60b170fc8027bb7898e2efca1"
        )
        self.assertEqual(config.source.qrels_members["test"], "qrels/test.tsv")


if __name__ == "__main__":
    unittest.main()
