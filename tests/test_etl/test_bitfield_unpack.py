"""Unit tests for the probe-attribute bitfield unpacker."""

from __future__ import annotations

from mongoose.etl.derived import unpack_attribute_bitfield
from mongoose.etl.schema import ATTRIBUTE_BIT_FIELDS


def test_all_zero_yields_all_false():
    bits = unpack_attribute_bitfield(0)
    assert all(v is False for v in bits.values())
    # And every declared field is present.
    for name, _ in ATTRIBUTE_BIT_FIELDS:
        assert name in bits


def test_accepted_only():
    bits = unpack_attribute_bitfield(1 << 7)
    assert bits["attr_accepted"] is True
    assert bits["attr_clean_region"] is False
    assert bits["attr_folded_end"] is False
    assert bits["attr_in_structure"] is False


def test_all_declared_bits_roundtrip():
    # Set every declared bit, confirm every named field comes back True.
    raw = 0
    for _, bit in ATTRIBUTE_BIT_FIELDS:
        raw |= 1 << bit
    bits = unpack_attribute_bitfield(raw)
    for name, _ in ATTRIBUTE_BIT_FIELDS:
        assert bits[name] is True


def test_high_bits_outside_declared_set_ignored():
    # Setting bit 20 (not in the spec) should not leak into any named field.
    raw = 1 << 20
    bits = unpack_attribute_bitfield(raw)
    assert all(v is False for v in bits.values())


def test_bit_positions_match_spec():
    """Documenting the canonical mapping; if someone reorders ATTRIBUTE_BIT_FIELDS
    this test will fail informatively."""
    mapping = dict(ATTRIBUTE_BIT_FIELDS)
    assert mapping["attr_clean_region"] == 0
    assert mapping["attr_folded_end"] == 1
    assert mapping["attr_folded_start"] == 2
    assert mapping["attr_in_structure"] == 3
    assert mapping["attr_excl_amp_high"] == 4
    assert mapping["attr_excl_width_sp"] == 5
    assert mapping["attr_excl_width_remap"] == 6
    assert mapping["attr_accepted"] == 7
    assert mapping["attr_excl_outside_partial"] == 8
