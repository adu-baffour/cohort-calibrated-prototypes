# Test bundle part: final evaluation round (drafted after validation; NOT run)

Run only by `python scripts/final_round.py bundle` after the human writes this block's token into `bundles/FINAL_APPROVED`. The token is the first 16 hex characters of the sha256 of the YAML block below. `--dry-run-val` runs the identical code path on the validation sets and requires every number to equal the committed validation outputs.
Decisions after this test: none.

```yaml-fsor-final-round
version: 1
registered: '2026-10-05'
part: final-round
approvable: true
status: dry run on validation passed (results/final_round/dryrun_val.json); approvable
code_commit: 9e3647e41d8c4db7f664b1269e727c88309e4751
preregistration:
  file: docs/preregistration/final_round.md
  sha256: e658be46e85d2a0b62d5696b61b5c8382dd4d3714bbcef1e78fc371a57bc8df0
validation_records:
  results/final_round/cifarfs_selection_valA.json: 0e8759dac652a9156619b9e44fef044e25509fb4512098324dc0e142ab4ffabf
  results/final_round/cifarfs_valB.json: 7f8dbf6f08439e1dd64310a6304cdcd1aa5b1a967c7f2a74071cabb3ce3a7262
  results/final_round/cub_val_preview.json: 51d7e8f79629d4aa0f6c4b770aa0df165ceffafe06dd099a3318b498d60a9f4a
  results/final_round/recorded_valA.json: 71cf234809a3c4b8095420b07580af5deca074452a7874dfcbd99c1ae18d88d1
  results/final_round/d1_val.json: 3e8d776f4b7e834654b5f2ed8d61459da2582790e4edbf31beda5d8f6fddae6c
test_manifests:
  mini/test_fsor_5way_1shot.json: 06e0e0022531b5ba91dd658dbe35ee62eb7ffe6540da1afc5450185f2982af05
  mini/test_fsor_5way_5shot.json: a8aa5f3718fa5a21cbd4c7a836974874a8c91ef5908abc544b9a4b07cfa57faa
  tiered/test_fsor_5way_1shot.json: 5f60e7b90a5cdbcb7ca79972f98d68552fc37238c728a39bda91d446c15fc390
  tiered/test_fsor_5way_5shot.json: c29cde57a0f1013885fd271a6ea9ddc7d40f553b13d9bfc42e5e1ad3ee46bc89
  cifarfs/test_fsor_5way_1shot.json: 8c232e3a8bf72a529562418ce6f9c230d6ac59580bc8d75d253c15e12dd5ab6d
  cifarfs/test_fsor_5way_5shot.json: 9a2c8c912fbcaa76c9de07cad23d707e4fdda79a32cd26f668f9ef3c8302779e
  cub/test_fsor_5way_1shot.json: 9ae7f4bff92ab9205e113d752024497aedbac3cf958f809c47806650910cda39
  cub/test_fsor_5way_5shot.json: bc8475d1c7c8a12e77bc378315c2acf44c9b17614fe9647f63a49d3636a80d8f
cifarfs_chosen:
  B1 QTERM: B1-QTERM
  B2 KNN: B2-KNN/k=1000/sign=-
  B3 RMD: B3-RMD/lambda=1.0
  B4 CL2N: B4-CL2N
  B5 ZN: B5-ZN
  B6 NNN: B6-NNN/alpha=0.25/k=1000
  B7a QB-IS: B7a-QB-IS/beta=10
  B7b QB-DIS: B7b-QB-DIS/beta=10
  B8 TEEN: B8-TEEN
  B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
cifarfs_weights:
  '1':
    B2-KNN/k=100/sign=-: 1.9685760114952484
    B2-KNN/k=10/sign=+: 1.9072642176658714
    B2-KNN/k=10/sign=-: 1.9072642176658714
    B2-KNN/k=100/sign=+: 1.9685760114952484
    B2-KNN/k=1000/sign=+: 2.8174501964978074
    B1-QTERM: 2.429415275385992
    B2-KNN/k=1000/sign=-: 2.8174501964978074
  '5':
    B2-KNN/k=100/sign=-: 1.9084844905502634
    B2-KNN/k=10/sign=+: 1.8532429435277602
    B2-KNN/k=10/sign=-: 1.8532429435277602
    B2-KNN/k=100/sign=+: 1.9084844905502634
    B2-KNN/k=1000/sign=+: 2.722112089672114
    B1-QTERM: 2.34402517351971
    B2-KNN/k=1000/sign=-: 2.722112089672114
hosts:
  cifarfs_rfs_seed0:
    dataset: cifarfs
    checkpoints:
      checkpoint:
        path: rfs/cifarfs/seed0/last.pth
        sha256: 69f7f2975bbac2c9ae299a3ec59bd0522f737b65a6d5012462741376889d31b9
    banks:
      all: da648a6ebb35a93effc53e881c365bfdfb95526f4131c83406452d8b12166ff4
  dnpg_mini_seed0:
    dataset: mini
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
    H_fix_weight:
      1shot: 2.626274957560938
      5shot: 2.888997703590982
    V1_weight:
      1shot: 2.54622457184192
      5shot: 2.8234188608120436
    cub_baselines:
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
    cub_target_bank:
      1shot: a9c46ccd3ff73af9a614f65fdc25a5715fc9bcdf2890f76b30d82f6aa55e8e3b
      5shot: 4a0205253e9db76235eb4c45426817379f03c66e6f5f2963a426e8bc7f7087ff
    recorded_test:
      1shot:
        host:
          auroc: 73.94856296296297
          acc: 69.08
        H_fix:
          auroc: 74.44755555555555
          acc: 69.08
        CCP-A:
          auroc: 73.62595555555555
          acc: 69.08
        COSF:
          auroc: 71.63398518518518
          acc: 69.08
      5shot:
        host:
          auroc: 83.46554074074074
          acc: 83.88888888888889
        H_fix:
          auroc: 84.00465185185186
          acc: 83.88888888888889
        CCP-A:
          auroc: 82.18165925925925
          acc: 83.88888888888889
        COSF:
          auroc: 77.81112592592594
          acc: 83.88888888888889
  dnpg_mini_seed1:
    dataset: mini
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
    H_fix_weight:
      1shot: 2.6399514523812333
      5shot: 2.912096116907742
    V1_weight:
      1shot: 2.5551429611557963
      5shot: 2.855901894852765
    cub_baselines:
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
    cub_target_bank:
      1shot: 7a7f710ff46a8191d7ee484d3ba20d9a484e5f497045d278ccfbd21357d151ec
      5shot: 911221c3f876326879eb75b9efdb4efc280cee1afefc0698d47f189dca02c851
    recorded_test:
      1shot:
        host:
          auroc: 73.57725925925925
          acc: 68.84
        H_fix:
          auroc: 74.22201481481481
          acc: 68.84
        CCP-A:
          auroc: 73.5239111111111
          acc: 68.84
        COSF:
          auroc: 71.56666666666666
          acc: 68.84
      5shot:
        host:
          auroc: 83.01428148148149
          acc: 83.90444444444445
        H_fix:
          auroc: 83.86162962962963
          acc: 83.90444444444445
        CCP-A:
          auroc: 82.27282962962963
          acc: 83.90444444444445
        COSF:
          auroc: 78.03765925925926
          acc: 83.90444444444445
  dnpg_mini_seed2:
    dataset: mini
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
    H_fix_weight:
      1shot: 2.6329064035074827
      5shot: 2.867463077549045
    V1_weight:
      1shot: 2.5392163509449883
      5shot: 2.8009096562955724
    cub_baselines:
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
    cub_target_bank:
      1shot: 7de389ff9cf6391bf5dafba75c78b5118fb5eb10e0d0645215f40f2c1afb7e5b
      5shot: 16904e8d065d5391c12400ba6b919a22b206e445aa1204c3183e4cafc67fa402
    recorded_test:
      1shot:
        host:
          auroc: 73.63511111111112
          acc: 68.8111111111111
        H_fix:
          auroc: 74.2389925925926
          acc: 68.8111111111111
        CCP-A:
          auroc: 73.37887407407408
          acc: 68.8111111111111
        COSF:
          auroc: 71.50177777777778
          acc: 68.8111111111111
      5shot:
        host:
          auroc: 82.89087407407408
          acc: 83.85333333333334
        H_fix:
          auroc: 83.684
          acc: 83.85333333333334
        CCP-A:
          auroc: 82.17217777777779
          acc: 83.85333333333334
        COSF:
          auroc: 77.88497777777778
          acc: 83.85333333333334
  frn_mini_seed0:
    dataset: mini
    checkpoints:
      checkpoint:
        path: frn/mini/seed0/best.pth
        sha256: 54d9569670ca0d38a183a57dc6bc209b169c270203a2850a5b4c51bbf69ba178
    banks:
      all: bf710b23a78d51660f4eb6432c6313a8a6f42b45d4805bc7d84953365ca3dfcc
    recorded_test:
      1shot:
        host:
          auroc: 69.3950074074074
          acc: 65.1777791728576
        COSF:
          auroc: 69.47438518518518
          acc: 65.2200014591217
        CCP-A:
          auroc: 70.58816296296295
          acc: 65.26889033118884
      5shot:
        host:
          auroc: 75.37468148148149
          acc: 80.35555750628312
        COSF:
          auroc: 75.4077925925926
          acc: 80.35777966181438
        CCP-A:
          auroc: 77.85511111111111
          acc: 80.43555738031864
  frn_mini_seed1:
    dataset: mini
    checkpoints:
      checkpoint:
        path: frn/mini/seed1/best.pth
        sha256: 37cae1a1da309c76a5f0ab963ddf1ecbddbff283753cfc585fe005dd2ffc2b36
    banks:
      all: ce57ecfeca2ea8b47b9433f41e7d8ef81817efc8d91d4ff211fcf65b556caf29
    recorded_test:
      1shot:
        host:
          auroc: 69.5903111111111
          acc: 65.0111125210921
        COSF:
          auroc: 69.67537777777778
          acc: 65.12666811048985
        CCP-A:
          auroc: 70.5354962962963
          acc: 64.98889022568862
      5shot:
        host:
          auroc: 75.81256296296296
          acc: 80.1288907925288
        COSF:
          auroc: 75.83626666666666
          acc: 80.2422239780426
        CCP-A:
          auroc: 77.81964444444445
          acc: 80.2000017662843
  frn_mini_seed2:
    dataset: mini
    checkpoints:
      checkpoint:
        path: frn/mini/seed2/best.pth
        sha256: 4eb7ceba7071397b5a29fbc6db7db6200c81985f25f6821dccb62df9d6fc99b0
    banks:
      all: 6c353711b48ebdd4e9f38c404905f12258d139e20694d9d3b9fb235d85954f50
    recorded_test:
      1shot:
        host:
          auroc: 69.69032592592592
          acc: 65.32889026403427
        COSF:
          auroc: 69.79514074074073
          acc: 65.37111248075962
        CCP-A:
          auroc: 70.76668148148147
          acc: 65.38444579641025
      5shot:
        host:
          auroc: 75.66142222222221
          acc: 80.30444633960724
        COSF:
          auroc: 75.69576296296297
          acc: 80.45555737614632
        CCP-A:
          auroc: 77.96373333333332
          acc: 80.35333521167438
  frozen_feat_mini:
    dataset: mini
    checkpoints:
      checkpoint:
        path: third_party/feat/mini/Res12-pre.pth
        sha256: 48e681c65e0a8a28d24f586f963da1f8b9e8aff339201f7380365bdd0ecc5c88
    banks:
      all: 829c3d9f7b45083c3a6cc9228879a9ff62cc3372c6127d5fcc423a6052bb271e
    cub_baselines:
      1shot:
        B1-QTERM: 2.586313779522386
        B2-KNN/k=1000/sign=-: 2.4780609179619657
        B3-RMD/lambda=1.0: null
        B4-CL2N: null
        B5-ZN: null
        B6-NNN/alpha=0.75/k=1000: null
        B7a-QB-IS/beta=10: null
        B7b-QB-DIS/beta=10: null
        B8-TEEN: null
        B9-CSLS-episode: null
      5shot:
        B1-QTERM: 2.525219800839733
        B2-KNN/k=1000/sign=-: 2.427045442003901
        B3-RMD/lambda=1.0: null
        B4-CL2N: null
        B5-ZN: null
        B6-NNN/alpha=0.75/k=1000: null
        B7a-QB-IS/beta=10: null
        B7b-QB-DIS/beta=10: null
        B8-TEEN: null
        B9-CSLS-episode: null
    cub_target_bank:
      1shot: 59d9d97c32bdaf046ac54cc92c1ffb210643a4dc35c7df37f0b8f6d4ec45ef6c
      5shot: 59d9d97c32bdaf046ac54cc92c1ffb210643a4dc35c7df37f0b8f6d4ec45ef6c
    recorded_test:
      1shot:
        host:
          auroc: 68.27007407407407
          acc: 63.282223592201866
        COSF:
          auroc: 68.41834074074073
          acc: 63.32666807373365
        CCP-A:
          auroc: 70.9021925925926
          acc: 64.94889022906621
      5shot:
        host:
          auroc: 74.64485925925928
          acc: 81.78889083862305
        COSF:
          auroc: 74.64217777777779
          acc: 82.01555745800336
        CCP-A:
          auroc: 78.88186666666665
          acc: 82.37111311157544
  frozen_feat_tiered:
    dataset: tiered
    checkpoints:
      checkpoint:
        path: third_party/feat/tiered/Res12-pre.pth
        sha256: 90bedb8ae0ad34946d2517a50d2636c0df52373fe5cc206d3a403516f5b51cf4
    banks:
      all: b8ba22dc3dba516ee7c41b2f9faf1af29db079b92e6a4fb357230e8f4641711d
    recorded_test:
      1shot:
        host:
          auroc: 73.9913925925926
          acc: 69.6800016015768
        COSF:
          auroc: 74.1373925925926
          acc: 70.18444596727689
        CCP-A:
          auroc: 74.5562962962963
          acc: 70.83111276229222
      5shot:
        host:
          auroc: 82.31788148148148
          acc: 85.27333545684814
        COSF:
          auroc: 82.43628148148149
          acc: 85.6644465426604
        CCP-A:
          auroc: 82.79428148148149
          acc: 85.71555763483047
  snatcher_f_mini:
    dataset: mini
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
    H_fix_weight:
      1shot: 39.18493949698625
      5shot: 47.22915457109874
    V1_weight:
      1shot: 38.76459974482703
      5shot: 46.819240085560914
    cub_baselines:
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
    cub_target_bank:
      1shot: 12151f0fe7465560761e0d6a2ad9d7ecb08ab73126b2f7be021af86f97c541a0
      5shot: 1bb2f2e3dde7b6211705901e83cba68dd5c380b3d2d01872765046ffc26fe601
    recorded_test:
      1shot:
        host:
          auroc: 69.88051851851851
          acc: 67.27555739720663
        H_fix:
          auroc: 71.55122962962962
          acc: 67.27555739720663
        CCP-A:
          auroc: 72.08114074074074
          acc: 67.14000156025092
        COSF:
          auroc: 69.58574814814814
          acc: 66.26889040569465
      5shot:
        host:
          auroc: 77.2817037037037
          acc: 82.77777959187826
        H_fix:
          auroc: 79.3839111111111
          acc: 82.77777959187826
        CCP-A:
          auroc: 80.0728888888889
          acc: 82.98889095584552
        COSF:
          auroc: 76.69519999999999
          acc: 83.06000195940335
  tane_pre_mini:
    dataset: mini
    checkpoints:
      checkpoint:
        path: third_party/tane/miniImageNet_pre.pth
        sha256: 2bd34c4594b77faa3710bca3f363f54219754951e24bc353be0c1b41f3d22bbb
    banks:
      all: 12d2c201b724f84770bedf3f097e6640163ea5d587f75dfa6a000991c47b555e
    recorded_test:
      1shot:
        host:
          auroc: 70.20967407407406
          acc: 65.4133348017931
        COSF:
          auroc: 70.37189629629628
          acc: 65.70889037350814
        CCP-A:
          auroc: 72.91628148148149
          acc: 67.10444590449333
      5shot:
        host:
          auroc: 76.744
          acc: 82.77111313243707
        COSF:
          auroc: 76.91640000000001
          acc: 83.03777972360452
        CCP-A:
          auroc: 81.35328888888888
          acc: 83.24666861693065
  tane_pre_tiered:
    dataset: tiered
    checkpoints:
      checkpoint:
        path: third_party/tane/tieredImageNet_pre.pth
        sha256: 93b39fc2ed47fff92bdf520a8c846be64375082c38d31effee94826eca7f4572
    banks:
      all: 360d8b9fff31a4278828cdd758fa94f4b39fb08543a78563262c286e23efe257
    recorded_test:
      1shot:
        host:
          auroc: 74.0642074074074
          acc: 69.40444594124953
        COSF:
          auroc: 74.26577777777777
          acc: 69.74889050920804
        CCP-A:
          auroc: 74.34494814814815
          acc: 70.03111273050308
      5shot:
        host:
          auroc: 81.84755555555554
          acc: 84.96000217398007
        COSF:
          auroc: 82.05571851851853
          acc: 85.19777993361156
        CCP-A:
          auroc: 82.31842962962963
          acc: 85.14666871229808
rows:
  A: 'CIFAR-FS: host (COSF), CCP-A, V1, the chosen B1-B9'
  B: 'CUB, constants from mini: host, CCP-A / H_fix, V1 / V1fused, the mini-chosen B1-B9, and the ANALYSIS rows CCP-A@target / H_fix@target (K = 1000) and CCP-A@target-K154 / H_fix@target-K154 (K = 154)'
  C: 'recorded hosts: host (COS / S / D), COSF, CCP-A, H_fix (trained), V1 / V1fused'
  D: D1 standardised gap on mini, tiered, CUB (frozen FEAT mini) and CIFAR-FS; figure vs the CCP-A - COSF AUROC gain
metrics: per episode AUROC, accuracy, OSCR, FPR95; mean and 95% CI; paired differences vs the host (and vs CCP-A for part A)
technical_checks: sha256 of every checkpoint, bank and manifest; part C must reproduce every recorded test AUROC and accuracy exactly, otherwise the pass is reported as a failed reproduction and the new
  metrics are not interpreted; no NaN
outputs:
  summary: results/final_round/test.json, test.md, d1_test.png
  per_episode: results/final_round/test_episodes.npz
decisions_after_test: none
```
