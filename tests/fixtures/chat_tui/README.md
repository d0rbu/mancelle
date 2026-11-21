# chat_tui checkpoint-metadata fixtures

Verbatim copies of real checkpoint metadata, pinned as ground truth for the chat TUI's
leaf-aware setup loader (`scripts/chat_tui/setup.py`):

- `meta_000749.json` — the standard-mixture conditioned-SFT leaf
  (`l22_gemma3pt_clustered_splice_cf_c1024_100m_370_2node_pr1407_v1/…/wl_9172c540310ba842/
  fc_5738db84…/standard_sft_conditioned_checkpoints/`, step 749). SFT-shaped meta: top-level
  `encoder_config`, no `user_config.encoder`, `pretrain_conditioning` re-publish block.
  Source file sha256: `5dcfeb3a651d95bce45dfd5604ffeb1750cd259779bb961d4bb98d93e08bf1c3`.
- `meta_026500.json` — the leaf's direct pretraining ancestor (same identity directory).
  Pretrain-shaped meta with the full `user_config.conditioning` block (stats binding, seed,
  prompt selection). Also serves as the pretrain-shape regression target for
  `load_chat_setup` ≡ `load_method_setup`.
- `meta_000749_distinct_selection.json` — a SYNTHETIC copy of the leaf fixture whose
  `user_config.feature_selection.mean` is 7.0 (`std` is paired at 7.0 because geometric
  sampling requires equality). It exists only to prove the C1 contract that leaf prompt
  selection does not leak from the ancestor's mean-12 conditioning block.

The bulk of these files is contract data (the 16,254-id whitelist and the canonical cluster
definition), which is exactly what the loaders must read — do not sanitize or truncate.
`tests/test_chat_tui/test_leaf_meta_shape.py` pins the shape facts the loader relies on;
if a future checkpoint generation changes shape, re-pin deliberately (new fixture + updated
assertions), never by loosening assertions in place.
