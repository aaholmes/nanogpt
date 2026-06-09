"""
Generative, layer-count-agnostic topology for phase2's asymmetric skip structure.

The record architecture is NOT a symmetric U-Net. It has:
  * one FORWARD skip: save activation at layer `skip_src`, inject it at layer
    `skip_dst` — and `skip_dst` drops its attention (the skip replaces it).
  * one BACKOUT: save activation at `backout_src`; later attention layers read
    that frozen state as input, and `backout_sign * backout_lambda * x_backout`
    is applied to the residual at the end.
  * auxiliary per-layer flags (paired-heads, value-embeds, key-offset) that are
    secondary and placed by an even-spread density rule.

`build_topology(L, params)` maps a small fixed-length parameter vector to a valid
topology dict for ANY depth L, so a BO can search over it. The DEFAULT params
reproduce the current L=11 CORE structure (skip 3->6, backout 7, attn-skip at 6)
exactly; auxiliary placements reproduce the legacy *density*, not exact membership
(the legacy sets {0,2,5,9} etc. were irregular hand-tuned choices).
"""

# Legacy L=11 reference (for nesting / validation)
LEGACY_L11 = dict(
    num_layers=11,
    skip_src=3, skip_dst=6,          # forward skip 3 -> 6; layer 6 drops attention
    backout_src=7, backout_enabled=True, backout_sign=-1,
    paired_layers={0, 2, 5, 9},
    ve_layers={1, 2, 8, 9, 10},
    key_offset_layers={3, 10},
)

# Default generative params — chosen so the CORE matches LEGACY_L11 at L=11.
DEFAULT_PARAMS = dict(
    skip_src_frac=0.30,      # 3/10
    skip_span_frac=0.30,     # span 3 -> dst 6
    backout_src_frac=0.70,   # 7/10
    backout_enabled=True,
    backout_sign=-1,
    paired_density=4 / 11,
    ve_density=5 / 11,
    key_offset_density=2 / 11,
)


def _even_spread(n_on, L):
    """Place n_on 'on' layers as evenly as possible across [0, L-1]."""
    if n_on <= 0:
        return set()
    if n_on >= L:
        return set(range(L))
    # evenly spaced indices, biased to include both ends lightly
    return {round(i * (L - 1) / (n_on - 1)) if n_on > 1 else (L - 1) // 2
            for i in range(n_on)}


def build_topology(L, params=None):
    """Return a topology dict for depth L from generative params.

    Keys: num_layers, skip_src, skip_dst (=attn_skip_layer), backout_src,
    backout_enabled, backout_sign, paired_layers, ve_layers, key_offset_layers,
    attn_layers (all layers except skip_dst).
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    last = L - 1

    # ---- Core skip/backout (the searched structure) ----
    skip_src = max(0, min(last - 1, round(p["skip_src_frac"] * last)))
    span     = max(1, round(p["skip_span_frac"] * last))
    skip_dst = min(last, skip_src + span)
    if skip_dst <= skip_src:                      # guarantee validity
        skip_dst = min(last, skip_src + 1)
    # attn-skip layer can't be 0 or last (need attention at the boundaries)
    skip_dst = max(1, min(last - 1, skip_dst))

    backout_src = max(0, min(last, round(p["backout_src_frac"] * last)))
    # backout should sit after the skip destination, like the legacy design
    backout_src = max(skip_dst + 1, backout_src) if skip_dst + 1 <= last else skip_dst

    # ---- Auxiliary placements (density-driven, scale with L) ----
    paired = _even_spread(round(p["paired_density"] * L), L) - {skip_dst}
    ve     = _even_spread(round(p["ve_density"] * L), L)
    keyoff = _even_spread(round(p["key_offset_density"] * L), L)

    attn_layers = [i for i in range(L) if i != skip_dst]

    return dict(
        num_layers=L,
        skip_src=skip_src,
        skip_dst=skip_dst,            # == attn_skip_layer
        backout_src=backout_src,
        backout_enabled=bool(p["backout_enabled"]),
        backout_sign=int(p["backout_sign"]),
        paired_layers=paired,
        ve_layers=ve,
        key_offset_layers=keyoff,
        attn_layers=attn_layers,
    )


def validate_topology(topo):
    """Raise AssertionError if a topology is structurally invalid."""
    L = topo["num_layers"]
    assert 0 <= topo["skip_src"] < topo["skip_dst"] <= L - 1, "bad skip src/dst"
    assert 1 <= topo["skip_dst"] <= L - 2, "attn-skip layer must be interior"
    assert 0 <= topo["backout_src"] <= L - 1, "bad backout src"
    assert topo["skip_dst"] not in topo["paired_layers"], "attn-skip layer can't be paired"
    assert len(topo["attn_layers"]) == L - 1, "exactly one attn-skip layer"
    return True


if __name__ == "__main__":
    # Self-test: default params at L=11 must reproduce the legacy CORE exactly.
    t = build_topology(11)
    validate_topology(t)
    core_ok = (t["skip_src"] == LEGACY_L11["skip_src"]
               and t["skip_dst"] == LEGACY_L11["skip_dst"]
               and t["backout_src"] == LEGACY_L11["backout_src"]
               and t["backout_enabled"] == LEGACY_L11["backout_enabled"]
               and t["backout_sign"] == LEGACY_L11["backout_sign"])
    print(f"L=11 core matches legacy: {core_ok}")
    print(f"  generated: skip {t['skip_src']}->{t['skip_dst']}  "
          f"backout {t['backout_src']} (sign {t['backout_sign']})")
    print(f"  legacy:    skip {LEGACY_L11['skip_src']}->{LEGACY_L11['skip_dst']}  "
          f"backout {LEGACY_L11['backout_src']}")
    print(f"  aux densities (paired/ve/keyoff): "
          f"{len(t['paired_layers'])}/{len(t['ve_layers'])}/{len(t['key_offset_layers'])} of {11}")
    print(f"    paired={sorted(t['paired_layers'])} ve={sorted(t['ve_layers'])} "
          f"keyoff={sorted(t['key_offset_layers'])}")
    print(f"    (legacy paired={sorted(LEGACY_L11['paired_layers'])} "
          f"ve={sorted(LEGACY_L11['ve_layers'])} — density reproduced, not membership)")
    assert core_ok, "default params must nest the legacy core"

    # Scaling demo across depths
    print("\nScaling across L:")
    for L in [8, 11, 14, 16]:
        t = build_topology(L)
        validate_topology(t)
        print(f"  L={L:2d}: skip {t['skip_src']:2d}->{t['skip_dst']:2d}  "
              f"backout {t['backout_src']:2d}  "
              f"paired={sorted(t['paired_layers'])}")
    print("\nAll topologies valid.")
