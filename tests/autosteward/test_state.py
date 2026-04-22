from autosteward.state import ParsedDirective


class TestParsedDirectivePayloads:
    def test_round_trip_payload_preserves_directive_fields(self):
        directive = ParsedDirective(
            armed=True,
            raw_directive="/as5",
            requested_hops=5,
            effective_hops=5,
            sanitized_message="",
            warnings=["clamped"],
        )

        payload = directive.to_payload()
        restored = ParsedDirective.from_payload(payload)

        assert restored == directive

    def test_from_payload_rejects_non_mapping(self):
        try:
            ParsedDirective.from_payload("/as5")
        except TypeError as exc:
            assert "mapping" in str(exc).lower()
        else:
            raise AssertionError("Expected TypeError for non-mapping payload")