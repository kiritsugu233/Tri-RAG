import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tri_rag_harness.pdctp_real_protocol import (
    PDCTPRealProtocolError,
    derive_fiqa_budget_grid,
    freeze_pdctp_real_protocol,
    load_pdctp_real_protocol_config,
)
from tri_rag_harness.utils import fingerprint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "pdctp_fiqa_real_protocol_freeze_v1.json"
SOURCE = ROOT / "artifacts" / "pdctp_fiqa_source_audit_v1" / "source_audit.json"
WITNESS = (
    ROOT
    / "artifacts"
    / "pdctp_fiqa_source_audit_v1"
    / "role_feasibility_witness.json"
)
POWER = ROOT / "artifacts" / "pdctp_network_free" / "power_plan_v1.json"
FREEZE_ROOT = ROOT / "artifacts" / "pdctp_fiqa_real_protocol_v1"


class PDCTPRealProtocolTests(unittest.TestCase):
    def test_checked_in_config_freezes_real_contract(self):
        config = load_pdctp_real_protocol_config(CONFIG)
        self.assertEqual(config.raw["retrieval"]["projection"]["m_prime"], 192)
        self.assertEqual(config.raw["retrieval"]["projection"]["seed"], 83047)
        self.assertEqual(config.feature_spec.lid_boundary, 32)
        self.assertEqual(config.budget_grid, derive_fiqa_budget_grid(57638, 64))
        self.assertEqual(config.budget_grid[-1], 57638)
        self.assertEqual(
            config.raw["candidate_suite"]["expected_full_pdctp_tuples"], 1620
        )
        self.assertEqual(
            config.raw["dataset"]["empty_documents"]["formatted_embedding_text"],
            "passage: [EMPTY_DOCUMENT]",
        )
        self.assertFalse(config.raw["stop_gates"]["method_evaluation_authorized"])
        self.assertEqual(
            config.config_fingerprint,
            "47c602c777e9e4589597ae996a7d1459407ae916b376854699569c115ebdfc41",
        )

    def test_freeze_is_reproducible_power_supported_and_all_roles_closed(self):
        config = load_pdctp_real_protocol_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            first = freeze_pdctp_real_protocol(
                config, SOURCE, WITNESS, POWER, directory / "first"
            )
            second = freeze_pdctp_real_protocol(
                config, SOURCE, WITNESS, POWER, directory / "second"
            )
            self.assertEqual(set(first), set(second))
            for name in first:
                self.assertEqual(first[name].read_bytes(), second[name].read_bytes())

            freeze = json.loads(first["protocol_freeze.json"].read_text())
            roles = json.loads(first["role_assignments.json"].read_text())
            state = json.loads(first["protocol_state.json"].read_text())
            self.assertEqual(freeze["decision"], "READY_FOR_DATASET_AND_EMBEDDING_AUDIT_ONLY")
            self.assertFalse(freeze["authorizes_method_evaluation"])
            self.assertFalse(freeze["authorizes_protected_outcome_access"])
            self.assertEqual(
                freeze["power_gate"],
                {"required_query_cert": 1567, "actual_query_cert": 1567, "passed": True},
            )
            self.assertEqual(
                {role: row["n"] for role, row in roles["roles"].items()},
                {
                    "query_cal": 1966,
                    "query_tune": 1967,
                    "query_cert": 1567,
                    "query_latency": 500,
                    "query_test": 648,
                },
            )
            all_ids = [
                query_id
                for role in roles["roles"].values()
                for query_id in role["ordered_ids"]
            ]
            self.assertEqual(len(all_ids), len(set(all_ids)))
            self.assertTrue(roles["all_roles_initially_closed"])
            for key in (
                "calibration_opened",
                "certification_opened",
                "latency_opened",
                "test_opened",
            ):
                self.assertFalse(state[key])
            self.assertIsNone(state["selection_fingerprint"])
            self.assertEqual(state["fit_artifacts"], {})

    def test_checked_in_freeze_artifacts_are_self_consistent(self):
        freeze_path = FREEZE_ROOT / "protocol_freeze.json"
        roles_path = FREEZE_ROOT / "role_assignments.json"
        state_path = FREEZE_ROOT / "protocol_state.json"
        freeze = json.loads(freeze_path.read_text())
        roles = json.loads(roles_path.read_text())
        state = json.loads(state_path.read_text())
        for artifact in (freeze, roles, state):
            body = dict(artifact)
            claimed = body.pop("fingerprint")
            self.assertEqual(claimed, fingerprint(body))
        self.assertEqual(
            freeze["fingerprint"],
            "cb3ef70f3ffc801c248f3269e0807480f0ee5a51cde41a52573a03f228a42368",
        )
        self.assertEqual(
            roles["fingerprint"],
            "ae884e0001d92ad11ddc1e420ece5412846454864331843a25a7e5cccf445dfe",
        )
        self.assertEqual(
            state["fingerprint"],
            "6f1eb5a4ecef7cf4a0413c13de82cd31dc2024b1593ebd86c57c153c276f45b9",
        )
        self.assertEqual(
            hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
            "a04a611b24da364dc294d65b857d017e84cab0802e4fc83dccb558858356ce8a",
        )
        self.assertFalse(roles["authorizes_outcome_access"])

    def test_tampered_input_is_rejected_before_output(self):
        config = load_pdctp_real_protocol_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            tampered = directory / "source.json"
            raw = json.loads(SOURCE.read_text())
            raw["counts"]["corpus"] += 1
            tampered.write_text(json.dumps(raw), encoding="utf-8")
            output = directory / "output"
            with self.assertRaisesRegex(PDCTPRealProtocolError, "SHA-256"):
                freeze_pdctp_real_protocol(config, tampered, WITNESS, POWER, output)
            self.assertFalse(output.exists())

    def test_mutations_of_projection_grid_roles_and_stop_gate_are_rejected(self):
        original = json.loads(CONFIG.read_text())
        mutations = []
        projection = copy.deepcopy(original)
        projection["retrieval"]["projection"]["m_prime"] = 191
        mutations.append((projection, "projection"))
        grid = copy.deepcopy(original)
        grid["retrieval"]["m_grid"][1] = 97
        mutations.append((grid, "budget grid"))
        roles = copy.deepcopy(original)
        roles["roles"]["counts"]["query_cert"] = 1566
        mutations.append((roles, "five-role counts"))
        stop = copy.deepcopy(original)
        stop["stop_gates"]["method_evaluation_authorized"] = True
        mutations.append((stop, "stop/go"))
        gpu = copy.deepcopy(original)
        gpu["latency"]["gpu_max_compatible_budget"] = 2048
        mutations.append((gpu, "latency execution"))
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            for index, (raw, message) in enumerate(mutations):
                path = directory / f"mutation-{index}.json"
                path.write_text(json.dumps(raw), encoding="utf-8")
                with self.subTest(index=index):
                    with self.assertRaisesRegex(PDCTPRealProtocolError, message):
                        load_pdctp_real_protocol_config(path)


if __name__ == "__main__":
    unittest.main()
