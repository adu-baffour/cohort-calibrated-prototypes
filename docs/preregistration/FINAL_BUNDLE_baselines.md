# Test bundle part: training-free baselines (drafted after validation; NOT run)

Run only by `python scripts/baselines.py bundle` after the human writes this block's token into `bundles/FINAL_APPROVED`. The token is the first 16 hex characters of the sha256 of the YAML block below. `--dry-run-val` runs the identical code path on validation A and B.
Decisions after this test: none. Its results will be reported whatever they are.

```yaml-fsor-baselines
version: 1
registered: '2026-09-30'
part: baselines
approvable: true
status: dry run on validation passed (results/baselines/dryrun_val.json); approvable
preregistration: docs/preregistration/baselines.md
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
chosen:
  frozen FEAT mini:
    B1 QTERM: B1-QTERM
    B2 KNN: B2-KNN/k=1000/sign=-
    B3 RMD: B3-RMD/lambda=1.0
    B4 CL2N: B4-CL2N
    B5 ZN: B5-ZN
    B6 NNN: B6-NNN/alpha=0.75/k=1000
    B7a QB-IS: B7a-QB-IS/beta=10
    B7b QB-DIS: B7b-QB-DIS/beta=10
    B8 TEEN: B8-TEEN
    B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
  frozen FEAT tiered:
    B1 QTERM: B1-QTERM
    B2 KNN: B2-KNN/k=1000/sign=-
    B3 RMD: B3-RMD/lambda=0.5
    B4 CL2N: B4-CL2N
    B5 ZN: B5-ZN
    B6 NNN: B6-NNN/alpha=0.75/k=1000
    B7a QB-IS: B7a-QB-IS/beta=20
    B7b QB-DIS: B7b-QB-DIS/beta=20
    B8 TEEN: B8-TEEN
    B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
  SnaTCHer-F mini:
    B1 QTERM: B1-QTERM
    B2 KNN: B2-KNN/k=100/sign=-
    B3 RMD: B3-RMD/lambda=0.5
    B4 CL2N: B4-CL2N
    B5 ZN: B5-ZN
    B6 NNN: B6-NNN/alpha=1.0/k=1000
    B7a QB-IS: B7a-QB-IS/beta=20
    B7b QB-DIS: B7b-QB-DIS/beta=20
    B8 TEEN: B8-TEEN
    B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
  DNPG mini:
    B1 QTERM: B1-QTERM
    B2 KNN: B2-KNN/k=10/sign=-
    B3 RMD: B3-RMD/lambda=0.5
    B4 CL2N: B4-CL2N
    B5 ZN: B5-ZN
    B6 NNN: B6-NNN/alpha=1.0/k=1000
    B7a QB-IS: B7a-QB-IS/beta=10
    B7b QB-DIS: B7b-QB-DIS/beta=10
    B8 TEEN: B8-TEEN
    B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
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
    weights:
      1shot:
        B1-QTERM: 2.586313779522386
        B2-KNN/k=1000/sign=-: 2.4780609179619657
      5shot:
        B1-QTERM: 2.525219800839733
        B2-KNN/k=1000/sign=-: 2.427045442003901
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
    weights:
      1shot:
        B1-QTERM: 2.887758386991465
        B2-KNN/k=1000/sign=-: 2.594877303908037
      5shot:
        B1-QTERM: 2.7716646702542413
        B2-KNN/k=1000/sign=-: 2.484494945583757
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
    weights:
      1shot:
        B1-QTERM: 96.53793134789889
        B2-KNN/k=100/sign=-: 77.90138021396162
        B3-RMD/lambda=0.5: 0.09267253202080605
        B4-CL2N: 19.311950332514964
        B5-ZN: 19.914794574030434
        B6-NNN/alpha=1.0/k=1000: 39.153217464662646
        B7a-QB-IS/beta=20: 1.9463995120006572
        B7b-QB-DIS/beta=20: 1.9463995120006572
        B8-TEEN: 38.856366418265786
        B9-CSLS-episode: 20.798971036849114
      5shot:
        B1-QTERM: 122.56676062642266
        B2-KNN/k=100/sign=-: 96.23274253008718
        B3-RMD/lambda=0.5: 0.12776548148897732
        B4-CL2N: 20.302692259986454
        B5-ZN: 20.982382250871964
        B6-NNN/alpha=1.0/k=1000: 46.67939112213889
        B7a-QB-IS/beta=20: 2.3024938720327364
        B7b-QB-DIS/beta=20: 2.3024938720327364
        B8-TEEN: 47.77774293078432
        B9-CSLS-episode: 25.245942297392713
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
    weights:
      1shot:
        B1-QTERM: 5.373996928561721
        B2-KNN/k=10/sign=-: 4.6573764858765125
        B3-RMD/lambda=0.5: 0.006069486644185038
        B4-CL2N: 1.4631443447070305
        B5-ZN: 1.520206894422496
        B6-NNN/alpha=1.0/k=1000: 2.672767178206897
        B7a-QB-IS/beta=10: 0.2504049879546468
        B7b-QB-DIS/beta=10: 0.2504049879546468
        B8-TEEN: 2.5547828237950836
        B9-CSLS-episode: 1.4081784321600055
      5shot:
        B1-QTERM: 5.8572633417904605
        B2-KNN/k=10/sign=-: 5.194968869555377
        B3-RMD/lambda=0.5: 0.007596849288568087
        B4-CL2N: 1.4351445348006335
        B5-ZN: 1.5269084916865283
        B6-NNN/alpha=1.0/k=1000: 2.8848499384542095
        B7a-QB-IS/beta=10: 0.26750543259354287
        B7b-QB-DIS/beta=10: 0.26750543259354287
        B8-TEEN: 2.8275429576584004
        B9-CSLS-episode: 1.5601003763848869
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
    weights:
      1shot:
        B1-QTERM: 5.3819016702654805
        B2-KNN/k=10/sign=-: 4.647704617613819
        B3-RMD/lambda=0.5: 0.006107512244327301
        B4-CL2N: 1.4740823524680902
        B5-ZN: 1.5320689224227195
        B6-NNN/alpha=1.0/k=1000: 2.691862245839057
        B7a-QB-IS/beta=10: 0.25225818797265814
        B7b-QB-DIS/beta=10: 0.25225818797265814
        B8-TEEN: 2.561879178113247
        B9-CSLS-episode: 1.4160826826412416
      5shot:
        B1-QTERM: 5.975879947174993
        B2-KNN/k=10/sign=-: 5.284661287103013
        B3-RMD/lambda=0.5: 0.007576978025079287
        B4-CL2N: 1.4343521832153832
        B5-ZN: 1.5253269045523625
        B6-NNN/alpha=1.0/k=1000: 2.891118266379392
        B7a-QB-IS/beta=10: 0.2685739823422404
        B7b-QB-DIS/beta=10: 0.2685739823422404
        B8-TEEN: 2.8682303873863937
        B9-CSLS-episode: 1.57527933416744
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
    weights:
      1shot:
        B1-QTERM: 5.246305350133973
        B2-KNN/k=10/sign=-: 4.539240322474417
        B3-RMD/lambda=0.5: 0.006100868290111521
        B4-CL2N: 1.4799946261847716
        B5-ZN: 1.5387716301223695
        B6-NNN/alpha=1.0/k=1000: 2.694264032023984
        B7a-QB-IS/beta=10: 0.2524796251556102
        B7b-QB-DIS/beta=10: 0.2524796251556102
        B8-TEEN: 2.537248125440631
        B9-CSLS-episode: 1.414848131710798
      5shot:
        B1-QTERM: 5.800705416975793
        B2-KNN/k=10/sign=-: 5.126345510307625
        B3-RMD/lambda=0.5: 0.0075166078042686576
        B4-CL2N: 1.424802541441664
        B5-ZN: 1.5154614972163432
        B6-NNN/alpha=1.0/k=1000: 2.8628735555745584
        B7a-QB-IS/beta=10: 0.26565678735340753
        B7b-QB-DIS/beta=10: 0.26565678735340753
        B8-TEEN: 2.808101206975859
        B9-CSLS-episode: 1.5522391419782509
rows: references host / COS / CCP and one row per chosen baseline (docs/preregistration/baselines.md, 'Reported rows')
outputs:
  summary: results/baselines/test.json, test.md
  per_episode: results/baselines/test_episodes.npz (every row, host, shot)
  paired: per-episode AUROC differences vs host and vs CCP with 95% CIs, Wilcoxon p, fraction of gains
decisions_after_test: none
```
