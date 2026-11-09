# CLIP Rosetta3 Strong-H1 follow-up

The complete experiment lives on branch `experiment/clip-strong-h1-swahili` of `ZijianZhang22/Multimodal_test_switch`.

This ZIP is an **overlay patch**, not a full repository checkout. Apply these files to `visual_anchor/` in the existing repo. The branch includes `prepare_swahili.py` and `test_swahili.py` copied from the earlier `experiment/swahili-rosetta3` branch, so the Swahili run works after GitHub checkout without manual copying.

- Chinese strong-H1 controls: `bash run_clip_followup.sh`
- Swahili repeat (frozen CLIP E/I): `bash run_swahili_clip.sh`
- Full instructions and design limitations: `visual_anchor/README_FOLLOWUP.md`

Requires cached CLIP features in `visual_anchor/real_work_clip_native/features.npz`.