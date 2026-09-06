from __future__ import annotations

from pathlib import Path

from gen_g8_d_handoff import build_handoff
from verify_g8_d_handoff import verify


def test_d7_handoff_binds_green_verification_and_g8e_boundary(
    tmp_path: Path, post_g10_am94
) -> None:
    del post_g10_am94
    output = tmp_path / "d7_handoff.json"
    artifact = build_handoff(pytest_count=2179)
    output.write_bytes(__import__("baseline.g8_d", fromlist=["rendered_json"]).rendered_json(artifact))
    verified = verify(output)
    assert verified["status"] == "GREEN"
    assert verified["next_gate"] == "G8_E/E0"
    assert verified["g8_e_released"] is True
    assert verified["full_campaign_not_started"] is True
