# Test bundle part: method variants (drafted after validation; NOT run)

Run only by `python scripts/variants.py bundle` after the human writes this block's token into `bundles/FINAL_APPROVED`. The token is the first 16 hex characters of the sha256 of the YAML block below. `--dry-run-val` runs the identical code path on valA, valB and valC and requires every number to equal the committed validation outputs.
Decisions after this test: none. Its results will be reported whatever they are.

```yaml-fsor-variants
version: 1
registered: '2026-10-02'
part: variants
approvable: true
status: dry run on validation passed (results/variants/dryrun_val.json); approvable
code_commit: 2edaa3fef72073e8cf4f2c5d67a89982d6dba6b2
preregistration:
  file: docs/prereg_variants.md
  sha256: c2ac6b59bd60d69e7a9bcf71e43d96da5cf49e6a174e95cbf9a051524bde7207
validation_records:
  results/variants/selection_valA.json: 59a56d79259340ed7d2c1c264b7cd9e2beb797772dc0498870611d98673cc9ef
  results/variants/validation_valB.json: b9cf5a5508186c33fa325d090b492bbc785123fef7aa85e892d03b34c5a332d5
  results/variants/validation_valC.json: ddaed66122c6d8a26957e4aaaaeda427a1e866b583c505a8a035ea1a3992b772
promotion:
  candidate_valB: V1 QW
  promoted: V1 QW
  setting: V1-QW/lambda=1.0/beta=0.5
  decision: V1 QW promoted (pre-registered extension; a test bundle is drafted for approval)
  G4_valC:
    mean_gain: 0.2988051851851852
    min_cell_gain: -0.03495999999999988
    G4: true
  rule: decided on validation before this test (docs/prereg_variants.md, Section 6); not revisited after it
chosen:
  V1 QW: V1-QW/lambda=1.0/beta=0.5
  V2 ASK: V2-ASK
  V3 SHC: V3-SHC/gamma=0.25
  V4 FW:
    SnaTCHer-F mini: V4-FW/mu=4
    DNPG mini: V4-FW/mu=1
episodes:
  mini:
    1shot:
      file: manifests/mini/test_fsor_5way_1shot.json
      sha256: 06e0e0022531b5ba91dd658dbe35ee62eb7ffe6540da1afc5450185f2982af05
    5shot:
      file: manifests/mini/test_fsor_5way_5shot.json
      sha256: a8aa5f3718fa5a21cbd4c7a836974874a8c91ef5908abc544b9a4b07cfa57faa
  tiered:
    1shot:
      file: manifests/tiered/test_fsor_5way_1shot.json
      sha256: 5f60e7b90a5cdbcb7ca79972f98d68552fc37238c728a39bda91d446c15fc390
    5shot:
      file: manifests/tiered/test_fsor_5way_5shot.json
      sha256: c29cde57a0f1013885fd271a6ea9ddc7d40f553b13d9bfc42e5e1ad3ee46bc89
hosts:
  frozen_feat_mini:
    dataset: mini
    group: frozen FEAT mini
    checkpoints:
      checkpoint:
        path: third_party/feat/mini/Res12-pre.pth
        sha256: 48e681c65e0a8a28d24f586f963da1f8b9e8aff339201f7380365bdd0ecc5c88
    banks:
      all: 829c3d9f7b45083c3a6cc9228879a9ff62cc3372c6127d5fcc423a6052bb271e
    ccp_weight_H_fix: null
    variant_weights: null
    recorded_REF_test:
      1shot:
        auroc: 70.9021925925926
        acc: 64.94889022906621
      5shot:
        auroc: 78.88186666666665
        acc: 82.37111311157544
  frozen_feat_tiered:
    dataset: tiered
    group: frozen FEAT tiered
    checkpoints:
      checkpoint:
        path: third_party/feat/tiered/Res12-pre.pth
        sha256: 90bedb8ae0ad34946d2517a50d2636c0df52373fe5cc206d3a403516f5b51cf4
    banks:
      all: b8ba22dc3dba516ee7c41b2f9faf1af29db079b92e6a4fb357230e8f4641711d
    ccp_weight_H_fix: null
    variant_weights: null
    recorded_REF_test:
      1shot:
        auroc: 74.5562962962963
        acc: 70.83111276229222
      5shot:
        auroc: 82.79428148148149
        acc: 85.71555763483047
  snatcher_f_mini:
    dataset: mini
    group: SnaTCHer-F mini
    checkpoints:
      1shot:
        path: third_party/feat_meta/mini/feat-1-shot.pth
        sha256: 60b78143149fb891f1b4d187fe24f03fb431d12d4c853752bf7dac5f90460a41
      5shot:
        path: third_party/feat_meta/mini/feat-5-shot.pth
        sha256: ee54213c754b2c25d07bb7b032549140d7be8d4e253f597cfa57a314696191ee
    banks:
      1shot: 09121a7ddcc38d4b3ba99a6d3437842de1007c00574276d88bdd93a44defe13c
      5shot: 0b849dde23919187033b2d7151e112cdcfb9ab6b9ec82e30983427ef6bb1d7be
    ccp_weight_H_fix:
      1shot: 39.18493949698625
      5shot: 47.22915457109874
    variant_weights:
      1shot:
        V1-QW/lambda=1.0/beta=0.5: 38.76459974482703
        V2-ASK: 1.4120667578118482
        V3-SHC/gamma=0.25: 34.1076720379299
      5shot:
        V1-QW/lambda=1.0/beta=0.5: 46.819240085560914
        V2-ASK: 1.5437144508634395
        V3-SHC/gamma=0.25: 40.16187584342237
    recorded_REF_test:
      1shot:
        auroc: 71.55122962962962
        acc: 67.27555739720663
      5shot:
        auroc: 79.3839111111111
        acc: 82.77777959187826
  dnpg_mini_seed0:
    dataset: mini
    group: DNPG mini
    checkpoints:
      phase2:
        path: dnpg/mini/seed0/phase2_mini_last.pth
        sha256: ecb712d1a868d4243556fbcb422fafb8520ced990d08e379389510006c095092
      max_acc_1shot:
        path: dnpg/mini/seed0/max_acc_1shot.pth
        sha256: ebf32fdbe474903584669530a9bb5a46a0ccebd4d59a9abd97ba734211f06d09
      max_acc_5shot:
        path: dnpg/mini/seed0/max_acc_5shot.pth
        sha256: 7e5e7b2b472bf6cf2bce2397156fddcc92d78286cc04aa578d88ddf2759fa674
    banks:
      1shot: 68c5ee17dd25af0de31ee759b2d365b745fab1974d94dddc84df853d57ac49d5
      5shot: 125c37ffd082eafbe73f581d09d2d9ba3a4a3e7b08431fb2947abaf05ebc5d1b
    ccp_weight_H_fix:
      1shot: 2.626274957560938
      5shot: 2.888997703590982
    variant_weights:
      1shot:
        V1-QW/lambda=1.0/beta=0.5: 2.54622457184192
        V2-ASK: 0.1051571848275943
        V3-SHC/gamma=0.25: 2.3283794368151396
      5shot:
        V1-QW/lambda=1.0/beta=0.5: 2.8234188608120436
        V2-ASK: 0.10857332669676796
        V3-SHC/gamma=0.25: 2.525847025511375
    recorded_REF_test:
      1shot:
        auroc: 74.44755555555555
        acc: 69.08
      5shot:
        auroc: 84.00465185185186
        acc: 83.88888888888889
  dnpg_mini_seed1:
    dataset: mini
    group: DNPG mini
    checkpoints:
      phase2:
        path: dnpg/mini/seed1/phase2_mini_last.pth
        sha256: 077560ab2a38b8233f759bb176668aacaf9cc3e0d01c7b0d15ec1237ada9a951
      max_acc_1shot:
        path: dnpg/mini/seed1/max_acc_1shot.pth
        sha256: 0c80ffdb6a7c0a3e4b6f0a57983373d23d69c75eaec19805049925c7b24ed85a
      max_acc_5shot:
        path: dnpg/mini/seed1/max_acc_5shot.pth
        sha256: 82e8fb64f9f2a41a768b5a55f5ed182ea68925e93e91c44812609a683820e0b0
    banks:
      1shot: d4d79ec0e7d0902fad36f10544be6a7151796c4cae25b6d26f5269a956ecd7d0
      5shot: c52a5d56bc6da9b832c8030386ecf2be4bf4f325abc0dc4bbc6d60786d7f7813
    ccp_weight_H_fix:
      1shot: 2.6399514523812333
      5shot: 2.912096116907742
    variant_weights:
      1shot:
        V1-QW/lambda=1.0/beta=0.5: 2.5551429611557963
        V2-ASK: 0.1061091070596559
        V3-SHC/gamma=0.25: 2.34122656025881
      5shot:
        V1-QW/lambda=1.0/beta=0.5: 2.855901894852765
        V2-ASK: 0.10807155722444295
        V3-SHC/gamma=0.25: 2.5441385386766027
    recorded_REF_test:
      1shot:
        auroc: 74.22201481481481
        acc: 68.84
      5shot:
        auroc: 83.86162962962963
        acc: 83.90444444444445
  dnpg_mini_seed2:
    dataset: mini
    group: DNPG mini
    checkpoints:
      phase2:
        path: dnpg/mini/seed2/phase2_mini_last.pth
        sha256: e2a00f5c6076789bd83469f617e0b48daf4eb6d6148b613ecb6511df9e4716f0
      max_acc_1shot:
        path: dnpg/mini/seed2/max_acc_1shot.pth
        sha256: 948fb2ed7c3df5a4f86d0334e6acfb08f0bef39ee8fe17711359174bce859594
      max_acc_5shot:
        path: dnpg/mini/seed2/max_acc_5shot.pth
        sha256: ad5f5878788c21de8794cb2e6d5108122563e06a8450cb4312011c3ef416efea
    banks:
      1shot: b7af0836dad48575efc4181d093a41bb32f9436501e77dd78aa8075c815dc662
      5shot: f23809124ad48bdcb599d885a5380467ba9c6bd80ab0e5c2b5936a4f91c826fa
    ccp_weight_H_fix:
      1shot: 2.6329064035074827
      5shot: 2.867463077549045
    variant_weights:
      1shot:
        V1-QW/lambda=1.0/beta=0.5: 2.5392163509449883
        V2-ASK: 0.10734933252366724
        V3-SHC/gamma=0.25: 2.3349493202009772
      5shot:
        V1-QW/lambda=1.0/beta=0.5: 2.8009096562955724
        V2-ASK: 0.10823553961229444
        V3-SHC/gamma=0.25: 2.504908851018435
    recorded_REF_test:
      1shot:
        auroc: 74.2389925925926
        acc: 68.8111111111111
      5shot:
        auroc: 83.684
        acc: 83.85333333333334
reference_check:
  recorded_files:
    results/recorded/test/FINAL_TEST_FSOR_MINI_DONE.json: 7a74ebc39e1ba2924c93e1b8bc0e3ef9acdd66a6df9dbe181a2b659cc53c3613
    results/recorded/test/FINAL_TEST_FSOR_TIERED_DONE.json: 14af0de34e3e241d2fd22c11ea4731d0532f3c88767130930b24dccfe5b3be24
    results/recorded/test/FINAL_TEST_FSOR_MINI-DNPG_DONE.json: 273986da0134b233c2030c36482b09b1cd092b3167a5358a88315cdd579451c9
    results/recorded/test/FINAL_TEST_FSOR_MINI-SNATCHER_DONE.json: 621fe66d074200e6aff2787d446144eb048da8e3d0c3e26dc91c55a7264dc65b
  rule: REF (CCP-A for frozen FEAT, H_fix with the frozen weight for trained hosts) must reproduce the recorded single-pass test AUROC and accuracy of every host and shot exactly (|difference| = 0). If
    not, the pass is reported as a failed reproduction and no variant number is interpreted.
rows: 'REF and the four variants at their valA settings (V1 QW, V2 ASK, V3 SHC on all 8 cells; V4 FW on the 4 trained-host cells); paired per-episode AUROC change vs REF (DNPG: averaged over seeds per episode),
  accuracy change on frozen FEAT'
outputs:
  summary: results/variants/test.json
  per_episode: results/variants/test_episodes.npz (every row, host, shot)
decisions_after_test: none. The test numbers of all four variants are reported whatever they are; CCP-A's recorded test results stay unchanged in the paper.
```
