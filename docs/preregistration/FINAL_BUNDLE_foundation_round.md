# Test bundle part: foundation-model round (drafted after validation; NOT run)

Run only by `python scripts/foundation_round.py bundle` after the human writes this block's token into `bundles/FINAL_APPROVED`. `--dry-run-val` runs the identical code path on the validation sets and requires every number to equal the committed records.
Decisions after this test: none.

```yaml-fsor-foundation-round
version: 1
registered: '2026-10-06'
part: foundation-round
approvable: true
status: dry run on validation passed (results/foundation_round/dryrun_val.json); approvable
code_commit: 20789f4a961fa9dd1cc1a40f580b7b1caa3da5bb
preregistration:
  file: docs/preregistration/foundation_round.md
  sha256: b76d7164afa9cf1c8405ab3fd90925e365f7070728c6a211a8288508c50419de
validation_records:
  results/foundation_round/weights.json: 92a46d7e1949087808ab28803f64a7ac77b43d34632bef8790f417da97a2567b
  results/foundation_round/banks_built.json: 152d5fc1efe55b3a182d7cd40e8595c5511827f4866ae6a4255311c18195340e
  results/foundation_round/sanity.json: 56f5fcfb4be0946d0db017ba5e25a0b959ebe61cc75ae0f0cf4a31662d44d766
  results/foundation_round/selection_valA.json: 7c451eb6f4abd7f2c32f027a3e4c1383b5d313b410b30f8a3d099632f44d1b7f
  results/foundation_round/validation_valB.json: 8b1cd45db51edf62550b37e2d621c5a78f1684f9393f2db00019b873ea356113
  results/foundation_round/d1_val.json: ae165b47994d357ab62cd58faed804030d716fd95e2282c103578f1db648ddae
test_manifests:
  mini/test_fsor_5way_1shot.json: 06e0e0022531b5ba91dd658dbe35ee62eb7ffe6540da1afc5450185f2982af05
  mini/test_fsor_5way_5shot.json: a8aa5f3718fa5a21cbd4c7a836974874a8c91ef5908abc544b9a4b07cfa57faa
  tiered/test_fsor_5way_1shot.json: 5f60e7b90a5cdbcb7ca79972f98d68552fc37238c728a39bda91d446c15fc390
  tiered/test_fsor_5way_5shot.json: c29cde57a0f1013885fd271a6ea9ddc7d40f553b13d9bfc42e5e1ad3ee46bc89
  cifarfs/test_fsor_5way_1shot.json: 8c232e3a8bf72a529562418ce6f9c230d6ac59580bc8d75d253c15e12dd5ab6d
  cifarfs/test_fsor_5way_5shot.json: 9a2c8c912fbcaa76c9de07cad23d707e4fdda79a32cd26f668f9ef3c8302779e
  cub/test_fsor_5way_1shot.json: 9ae7f4bff92ab9205e113d752024497aedbac3cf958f809c47806650910cda39
  cub/test_fsor_5way_5shot.json: bc8475d1c7c8a12e77bc378315c2acf44c9b17614fe9647f63a49d3636a80d8f
hosts:
  mini_clip_b16:
    dataset: mini
    view: eval84
    weights:
      path: third_party/clip/ViT-B-16.pt
      sha256: 5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f
    bank: f1d5a301f2ff58d82ef879b6d58a62dc91e13244179ac77a43efb6c366b70a0f
    batch: 500
    chosen:
      B1 QTERM: B1-QTERM
      B2 KNN: B2-KNN/k=1000/sign=-
      B3 RMD: B3-RMD/lambda=1.0
      B4 CL2N: B4-CL2N
      B5 ZN: B5-ZN
      B6 NNN: B6-NNN/alpha=0.75/k=1000
      B7a QB-IS: B7a-QB-IS/beta=20
      B7b QB-DIS: B7b-QB-DIS/beta=20
      B8 TEEN: B8-TEEN
      B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
    evidence_weights:
      '1':
        B1-QTERM: 1.5700302571754343
        B2-KNN/k=100/sign=+: 1.5529576910868357
        B2-KNN/k=1000/sign=+: 1.522204720729034
        B2-KNN/k=100/sign=-: 1.5529576910868357
        B2-KNN/k=1000/sign=-: 1.522204720729034
        B2-KNN/k=10/sign=+: 1.627637653516437
        B2-KNN/k=10/sign=-: 1.627637653516437
      '5':
        B1-QTERM: 1.539165289251356
        B2-KNN/k=100/sign=+: 1.5206591689360143
        B2-KNN/k=1000/sign=+: 1.4958246054499502
        B2-KNN/k=100/sign=-: 1.5206591689360143
        B2-KNN/k=1000/sign=-: 1.4958246054499502
        B2-KNN/k=10/sign=+: 1.5959862421718405
        B2-KNN/k=10/sign=-: 1.5959862421718405
  tiered_clip_b16:
    dataset: tiered
    view: bgr84
    weights:
      path: third_party/clip/ViT-B-16.pt
      sha256: 5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f
    bank: 48e656e3d0c5bbb24d02d32f54da0b2fae9734a311eb86dfb2e992f2ae82be02
    batch: 500
    chosen:
      B1 QTERM: B1-QTERM
      B2 KNN: B2-KNN/k=100/sign=-
      B3 RMD: B3-RMD/lambda=0.5
      B4 CL2N: B4-CL2N
      B5 ZN: B5-ZN
      B6 NNN: B6-NNN/alpha=0.75/k=100
      B7a QB-IS: B7a-QB-IS/beta=50
      B7b QB-DIS: B7b-QB-DIS/beta=50
      B8 TEEN: B8-TEEN
      B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
    evidence_weights:
      '1':
        B1-QTERM: 1.312139891094691
        B2-KNN/k=100/sign=+: 1.368156371292697
        B2-KNN/k=1000/sign=+: 1.2791586973805864
        B2-KNN/k=100/sign=-: 1.368156371292697
        B2-KNN/k=1000/sign=-: 1.2791586973805864
        B2-KNN/k=10/sign=+: 1.4914035413331022
        B2-KNN/k=10/sign=-: 1.4914035413331022
      '5':
        B1-QTERM: 1.22668404554039
        B2-KNN/k=100/sign=+: 1.2797211575657037
        B2-KNN/k=1000/sign=+: 1.1952410288280775
        B2-KNN/k=100/sign=-: 1.2797211575657037
        B2-KNN/k=1000/sign=-: 1.1952410288280775
        B2-KNN/k=10/sign=+: 1.3953700030962928
        B2-KNN/k=10/sign=-: 1.3953700030962928
  cifarfs_clip_b16:
    dataset: cifarfs
    view: eval32
    weights:
      path: third_party/clip/ViT-B-16.pt
      sha256: 5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f
    bank: bd34fc64a14bfd23caa460ac9eb5f2139828e2f32dcf0d7871cc81fc6054b8ef
    batch: 500
    chosen:
      B1 QTERM: B1-QTERM
      B2 KNN: B2-KNN/k=1000/sign=-
      B3 RMD: B3-RMD/lambda=1.0
      B4 CL2N: B4-CL2N
      B5 ZN: B5-ZN
      B6 NNN: B6-NNN/alpha=0.75/k=1000
      B7a QB-IS: B7a-QB-IS/beta=20
      B7b QB-DIS: B7b-QB-DIS/beta=20
      B8 TEEN: B8-TEEN
      B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
    evidence_weights:
      '1':
        B1-QTERM: 1.2768819331019665
        B2-KNN/k=100/sign=+: 1.3456787239598764
        B2-KNN/k=1000/sign=+: 1.2242555305054983
        B2-KNN/k=100/sign=-: 1.3456787239598764
        B2-KNN/k=1000/sign=-: 1.2242555305054983
        B2-KNN/k=10/sign=+: 1.5311721482262892
        B2-KNN/k=10/sign=-: 1.5311721482262892
      '5':
        B1-QTERM: 1.1610546980179788
        B2-KNN/k=100/sign=+: 1.223556348862418
        B2-KNN/k=1000/sign=+: 1.1134365963731974
        B2-KNN/k=100/sign=-: 1.223556348862418
        B2-KNN/k=1000/sign=-: 1.1134365963731974
        B2-KNN/k=10/sign=+: 1.3919168969120836
        B2-KNN/k=10/sign=-: 1.3919168969120836
  cub_clip_b16:
    dataset: cub
    view: eval84
    weights:
      path: third_party/clip/ViT-B-16.pt
      sha256: 5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f
    bank: 5df6df8d4fd63ea2fecaec225263154ce97768c94481f25f08a2d53c8480b89e
    batch: 500
    chosen:
      B1 QTERM: B1-QTERM
      B2 KNN: B2-KNN/k=100/sign=-
      B3 RMD: B3-RMD/lambda=1.0
      B4 CL2N: B4-CL2N
      B5 ZN: B5-ZN
      B6 NNN: B6-NNN/alpha=0.75/k=1000
      B7a QB-IS: B7a-QB-IS/beta=50
      B7b QB-DIS: B7b-QB-DIS/beta=50
      B8 TEEN: B8-TEEN
      B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
    evidence_weights:
      '1':
        B1-QTERM: 1.744488662645306
        B2-KNN/k=100/sign=+: 1.8411846151081503
        B2-KNN/k=1000/sign=+: 1.607681059532317
        B2-KNN/k=100/sign=-: 1.8411846151081503
        B2-KNN/k=1000/sign=-: 1.607681059532317
        B2-KNN/k=10/sign=+: 2.007672892332983
        B2-KNN/k=10/sign=-: 2.007672892332983
      '5':
        B1-QTERM: 1.6582996518406339
        B2-KNN/k=100/sign=+: 1.7529691807639873
        B2-KNN/k=1000/sign=+: 1.5252785586756683
        B2-KNN/k=100/sign=-: 1.7529691807639873
        B2-KNN/k=1000/sign=-: 1.5252785586756683
        B2-KNN/k=10/sign=+: 1.9125345318847038
        B2-KNN/k=10/sign=-: 1.9125345318847038
  mini_dinov2_s14:
    dataset: mini
    view: eval84
    weights:
      path: third_party/dinov2/dinov2_vits14_pretrain.pth
      sha256: b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9
    bank: 5745c90fb1853d3da1f46efe1e950b22ced90945dff706cd3e9b7f6e0553ef17
    batch: 500
    chosen:
      B1 QTERM: B1-QTERM
      B2 KNN: B2-KNN/k=100/sign=-
      B3 RMD: B3-RMD/lambda=1.0
      B4 CL2N: B4-CL2N
      B5 ZN: B5-ZN
      B6 NNN: B6-NNN/alpha=0.25/k=1000
      B7a QB-IS: B7a-QB-IS/beta=10
      B7b QB-DIS: B7b-QB-DIS/beta=10
      B8 TEEN: B8-TEEN
      B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
    evidence_weights:
      '1':
        B1-QTERM: 2.467127544720618
        B2-KNN/k=100/sign=+: 1.4386701107471342
        B2-KNN/k=1000/sign=+: 4.721196984346642
        B2-KNN/k=100/sign=-: 1.4386701107471342
        B2-KNN/k=1000/sign=-: 4.721196984346642
        B2-KNN/k=10/sign=+: 1.364771121280009
        B2-KNN/k=10/sign=-: 1.364771121280009
      '5':
        B1-QTERM: 3.141384034995395
        B2-KNN/k=100/sign=+: 1.8461851363643562
        B2-KNN/k=1000/sign=+: 5.972311196005744
        B2-KNN/k=100/sign=-: 1.8461851363643562
        B2-KNN/k=1000/sign=-: 5.972311196005744
        B2-KNN/k=10/sign=+: 1.7502238705598883
        B2-KNN/k=10/sign=-: 1.7502238705598883
  tiered_dinov2_s14:
    dataset: tiered
    view: bgr84
    weights:
      path: third_party/dinov2/dinov2_vits14_pretrain.pth
      sha256: b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9
    bank: d2ee36e3a5197afe842b7cd7bafc822cc7ac66ad5b371beea8bc1eb764a4bb1e
    batch: 500
    chosen:
      B1 QTERM: B1-QTERM
      B2 KNN: B2-KNN/k=10/sign=-
      B3 RMD: B3-RMD/lambda=1.0
      B4 CL2N: B4-CL2N
      B5 ZN: B5-ZN
      B6 NNN: B6-NNN/alpha=0.5/k=100
      B7a QB-IS: B7a-QB-IS/beta=10
      B7b QB-DIS: B7b-QB-DIS/beta=10
      B8 TEEN: B8-TEEN
      B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
    evidence_weights:
      '1':
        B1-QTERM: 3.390346534778955
        B2-KNN/k=100/sign=+: 2.545715986230231
        B2-KNN/k=1000/sign=+: 4.0825906483701475
        B2-KNN/k=100/sign=-: 2.545715986230231
        B2-KNN/k=1000/sign=-: 4.0825906483701475
        B2-KNN/k=10/sign=+: 1.9773045072110544
        B2-KNN/k=10/sign=-: 1.9773045072110544
      '5':
        B1-QTERM: 4.0055874098875766
        B2-KNN/k=100/sign=+: 3.014305360801124
        B2-KNN/k=1000/sign=+: 4.81499893818327
        B2-KNN/k=100/sign=-: 3.014305360801124
        B2-KNN/k=1000/sign=-: 4.81499893818327
        B2-KNN/k=10/sign=+: 2.3421367724172937
        B2-KNN/k=10/sign=-: 2.3421367724172937
  cifarfs_dinov2_s14:
    dataset: cifarfs
    view: eval32
    weights:
      path: third_party/dinov2/dinov2_vits14_pretrain.pth
      sha256: b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9
    bank: 96d3592b0ffae1fea2750d1aa0f5cb812f7d241eded3f09b79b01feb0e26760c
    batch: 500
    chosen:
      B1 QTERM: B1-QTERM
      B2 KNN: B2-KNN/k=100/sign=-
      B3 RMD: B3-RMD/lambda=1.0
      B4 CL2N: B4-CL2N
      B5 ZN: B5-ZN
      B6 NNN: B6-NNN/alpha=0.5/k=1000
      B7a QB-IS: B7a-QB-IS/beta=10
      B7b QB-DIS: B7b-QB-DIS/beta=10
      B8 TEEN: B8-TEEN
      B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
    evidence_weights:
      '1':
        B1-QTERM: 1.9617400579751532
        B2-KNN/k=100/sign=+: 1.6485173886807702
        B2-KNN/k=1000/sign=+: 2.2300676271340296
        B2-KNN/k=100/sign=-: 1.6485173886807702
        B2-KNN/k=1000/sign=-: 2.2300676271340296
        B2-KNN/k=10/sign=+: 1.5345364862076065
        B2-KNN/k=10/sign=-: 1.5345364862076065
      '5':
        B1-QTERM: 2.358311328079175
        B2-KNN/k=100/sign=+: 1.9868609371694999
        B2-KNN/k=1000/sign=+: 2.6782429370918663
        B2-KNN/k=100/sign=-: 1.9868609371694999
        B2-KNN/k=1000/sign=-: 2.6782429370918663
        B2-KNN/k=10/sign=+: 1.8522775185327778
        B2-KNN/k=10/sign=-: 1.8522775185327778
  cub_dinov2_s14:
    dataset: cub
    view: eval84
    weights:
      path: third_party/dinov2/dinov2_vits14_pretrain.pth
      sha256: b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9
    bank: 2e3f71eecbef45a705ea635bb3bc0e6fc7b746c5cb4fdb177364c09ee45e6447
    batch: 500
    chosen:
      B1 QTERM: B1-QTERM
      B2 KNN: B2-KNN/k=100/sign=-
      B3 RMD: B3-RMD/lambda=1.0
      B4 CL2N: B4-CL2N
      B5 ZN: B5-ZN
      B6 NNN: B6-NNN/alpha=1.0/k=1000
      B7a QB-IS: B7a-QB-IS/beta=10
      B7b QB-DIS: B7b-QB-DIS/beta=10
      B8 TEEN: B8-TEEN
      B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
    evidence_weights:
      '1':
        B1-QTERM: 3.2056050877992814
        B2-KNN/k=100/sign=+: 2.6610665285258643
        B2-KNN/k=1000/sign=+: 3.325948042268856
        B2-KNN/k=100/sign=-: 2.6610665285258643
        B2-KNN/k=1000/sign=-: 3.325948042268856
        B2-KNN/k=10/sign=+: 2.5150623193159043
        B2-KNN/k=10/sign=-: 2.5150623193159043
      '5':
        B1-QTERM: 3.7299351629546305
        B2-KNN/k=100/sign=+: 3.0756861048790176
        B2-KNN/k=1000/sign=+: 3.852355891932883
        B2-KNN/k=100/sign=-: 3.0756861048790176
        B2-KNN/k=1000/sign=-: 3.852355891932883
        B2-KNN/k=10/sign=+: 2.9148774335371797
        B2-KNN/k=10/sign=-: 2.9148774335371797
  mini_dinov2_b14:
    dataset: mini
    view: eval84
    weights:
      path: third_party/dinov2/dinov2_vitb14_pretrain.pth
      sha256: 0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73
    bank: 0b083f036036bf359a21d7a69699582840e619fd02089d3541198e91e8c29c73
    batch: 500
    chosen:
      B1 QTERM: B1-QTERM
      B2 KNN: B2-KNN/k=1000/sign=+
      B3 RMD: B3-RMD/lambda=1.0
      B4 CL2N: B4-CL2N
      B5 ZN: B5-ZN
      B6 NNN: B6-NNN/alpha=0.25/k=1000
      B7a QB-IS: B7a-QB-IS/beta=10
      B7b QB-DIS: B7b-QB-DIS/beta=10
      B8 TEEN: B8-TEEN
      B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
    evidence_weights:
      '1':
        B1-QTERM: 3.1193780858086924
        B2-KNN/k=100/sign=+: 1.4757118497134791
        B2-KNN/k=1000/sign=+: 9.742739858438176
        B2-KNN/k=100/sign=-: 1.4757118497134791
        B2-KNN/k=1000/sign=-: 9.742739858438176
        B2-KNN/k=10/sign=+: 1.3234144081827144
        B2-KNN/k=10/sign=-: 1.3234144081827144
      '5':
        B1-QTERM: 3.832197346053881
        B2-KNN/k=100/sign=+: 1.8297537543115627
        B2-KNN/k=1000/sign=+: 11.86715598364048
        B2-KNN/k=100/sign=-: 1.8297537543115627
        B2-KNN/k=1000/sign=-: 11.86715598364048
        B2-KNN/k=10/sign=+: 1.6375994098745592
        B2-KNN/k=10/sign=-: 1.6375994098745592
  tiered_dinov2_b14:
    dataset: tiered
    view: bgr84
    weights:
      path: third_party/dinov2/dinov2_vitb14_pretrain.pth
      sha256: 0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73
    bank: 6f24b10e9599530926f06e6a6860fa19962a8ac218f35f33d2d0527e7f8a1d95
    batch: 500
    chosen:
      B1 QTERM: B1-QTERM
      B2 KNN: B2-KNN/k=10/sign=-
      B3 RMD: B3-RMD/lambda=1.0
      B4 CL2N: B4-CL2N
      B5 ZN: B5-ZN
      B6 NNN: B6-NNN/alpha=0.25/k=100
      B7a QB-IS: B7a-QB-IS/beta=10
      B7b QB-DIS: B7b-QB-DIS/beta=10
      B8 TEEN: B8-TEEN
      B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
    evidence_weights:
      '1':
        B1-QTERM: 5.210850747057045
        B2-KNN/k=100/sign=+: 3.2995195566561484
        B2-KNN/k=1000/sign=+: 7.57205009571364
        B2-KNN/k=100/sign=-: 3.2995195566561484
        B2-KNN/k=1000/sign=-: 7.57205009571364
        B2-KNN/k=10/sign=+: 1.9945639914908893
        B2-KNN/k=10/sign=-: 1.9945639914908893
      '5':
        B1-QTERM: 6.191799104191971
        B2-KNN/k=100/sign=+: 3.9311079693319475
        B2-KNN/k=1000/sign=+: 8.956602671045962
        B2-KNN/k=100/sign=-: 3.9311079693319475
        B2-KNN/k=1000/sign=-: 8.956602671045962
        B2-KNN/k=10/sign=+: 2.3710489509389157
        B2-KNN/k=10/sign=-: 2.3710489509389157
  cifarfs_dinov2_b14:
    dataset: cifarfs
    view: eval32
    weights:
      path: third_party/dinov2/dinov2_vitb14_pretrain.pth
      sha256: 0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73
    bank: 1c7b995ed21fa0b44835d4ce0bffecb79b648367c9596e99428671a048e9b190
    batch: 500
    chosen:
      B1 QTERM: B1-QTERM
      B2 KNN: B2-KNN/k=100/sign=-
      B3 RMD: B3-RMD/lambda=1.0
      B4 CL2N: B4-CL2N
      B5 ZN: B5-ZN
      B6 NNN: B6-NNN/alpha=0.25/k=1000
      B7a QB-IS: B7a-QB-IS/beta=10
      B7b QB-DIS: B7b-QB-DIS/beta=10
      B8 TEEN: B8-TEEN
      B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
    evidence_weights:
      '1':
        B1-QTERM: 2.3194339413660385
        B2-KNN/k=100/sign=+: 1.7844561239111976
        B2-KNN/k=1000/sign=+: 2.870616412275076
        B2-KNN/k=100/sign=-: 1.7844561239111976
        B2-KNN/k=1000/sign=-: 2.870616412275076
        B2-KNN/k=10/sign=+: 1.5569496617461094
        B2-KNN/k=10/sign=-: 1.5569496617461094
      '5':
        B1-QTERM: 2.88356249963208
        B2-KNN/k=100/sign=+: 2.222094100383239
        B2-KNN/k=1000/sign=+: 3.567540593299199
        B2-KNN/k=100/sign=-: 2.222094100383239
        B2-KNN/k=1000/sign=-: 3.567540593299199
        B2-KNN/k=10/sign=+: 1.9421843274638877
        B2-KNN/k=10/sign=-: 1.9421843274638877
  cub_dinov2_b14:
    dataset: cub
    view: eval84
    weights:
      path: third_party/dinov2/dinov2_vitb14_pretrain.pth
      sha256: 0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73
    bank: 588b7f412e8d562b687292df4d826a395c1e1a10d7c78c761c7d58f48db93277
    batch: 500
    chosen:
      B1 QTERM: B1-QTERM
      B2 KNN: B2-KNN/k=100/sign=-
      B3 RMD: B3-RMD/lambda=1.0
      B4 CL2N: B4-CL2N
      B5 ZN: B5-ZN
      B6 NNN: B6-NNN/alpha=0.75/k=1000
      B7a QB-IS: B7a-QB-IS/beta=10
      B7b QB-DIS: B7b-QB-DIS/beta=10
      B8 TEEN: B8-TEEN
      B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
    evidence_weights:
      '1':
        B1-QTERM: 3.6708429587856632
        B2-KNN/k=100/sign=+: 2.192764029446489
        B2-KNN/k=1000/sign=+: 4.745945100613368
        B2-KNN/k=100/sign=-: 2.192764029446489
        B2-KNN/k=1000/sign=-: 4.745945100613368
        B2-KNN/k=10/sign=+: 2.313251845720809
        B2-KNN/k=10/sign=-: 2.313251845720809
      '5':
        B1-QTERM: 4.177462939652268
        B2-KNN/k=100/sign=+: 2.4715354411714983
        B2-KNN/k=1000/sign=+: 5.381575321014327
        B2-KNN/k=100/sign=-: 2.4715354411714983
        B2-KNN/k=1000/sign=-: 5.381575321014327
        B2-KNN/k=10/sign=+: 2.619275268340252
        B2-KNN/k=10/sign=-: 2.619275268340252
  mini_dinov2_l14:
    dataset: mini
    view: eval84
    weights:
      path: third_party/dinov2/dinov2_vitl14_pretrain.pth
      sha256: d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428
    bank: 36a08d78e1b7e5f57cd72789854ef54a13f43137cd442f4b29f8b2a764a2277d
    batch: 500
    chosen:
      B1 QTERM: B1-QTERM
      B2 KNN: B2-KNN/k=100/sign=+
      B3 RMD: B3-RMD/lambda=1.0
      B4 CL2N: B4-CL2N
      B5 ZN: B5-ZN
      B6 NNN: B6-NNN/alpha=0.25/k=1000
      B7a QB-IS: B7a-QB-IS/beta=10
      B7b QB-DIS: B7b-QB-DIS/beta=10
      B8 TEEN: B8-TEEN
      B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
    evidence_weights:
      '1':
        B1-QTERM: 4.044484851460806
        B2-KNN/k=100/sign=+: 1.620696157554347
        B2-KNN/k=1000/sign=+: 18.017413371672014
        B2-KNN/k=100/sign=-: 1.620696157554347
        B2-KNN/k=1000/sign=-: 18.017413371672014
        B2-KNN/k=10/sign=+: 1.3542242637772837
        B2-KNN/k=10/sign=-: 1.3542242637772837
      '5':
        B1-QTERM: 4.70232672504263
        B2-KNN/k=100/sign=+: 1.9028425989564084
        B2-KNN/k=1000/sign=+: 20.479530438723053
        B2-KNN/k=100/sign=-: 1.9028425989564084
        B2-KNN/k=1000/sign=-: 20.479530438723053
        B2-KNN/k=10/sign=+: 1.5799684039626756
        B2-KNN/k=10/sign=-: 1.5799684039626756
  tiered_dinov2_l14:
    dataset: tiered
    view: bgr84
    weights:
      path: third_party/dinov2/dinov2_vitl14_pretrain.pth
      sha256: d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428
    bank: 8cc74be614dda05ce8084c12f828f3e0a2509c25d9c98de69b13df9b32829944
    batch: 500
    chosen:
      B1 QTERM: B1-QTERM
      B2 KNN: B2-KNN/k=1000/sign=+
      B3 RMD: B3-RMD/lambda=1.0
      B4 CL2N: B4-CL2N
      B5 ZN: B5-ZN
      B6 NNN: B6-NNN/alpha=0.5/k=1000
      B7a QB-IS: B7a-QB-IS/beta=10
      B7b QB-DIS: B7b-QB-DIS/beta=10
      B8 TEEN: B8-TEEN
      B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
    evidence_weights:
      '1':
        B1-QTERM: 6.915479855179108
        B2-KNN/k=100/sign=+: 3.970860468325197
        B2-KNN/k=1000/sign=+: 11.012964890973425
        B2-KNN/k=100/sign=-: 3.970860468325197
        B2-KNN/k=1000/sign=-: 11.012964890973425
        B2-KNN/k=10/sign=+: 2.100152735056454
        B2-KNN/k=10/sign=-: 2.100152735056454
      '5':
        B1-QTERM: 8.067673151675269
        B2-KNN/k=100/sign=+: 4.63766804971364
        B2-KNN/k=1000/sign=+: 12.826166460814749
        B2-KNN/k=100/sign=-: 4.63766804971364
        B2-KNN/k=1000/sign=-: 12.826166460814749
        B2-KNN/k=10/sign=+: 2.4518597965830606
        B2-KNN/k=10/sign=-: 2.4518597965830606
  cifarfs_dinov2_l14:
    dataset: cifarfs
    view: eval32
    weights:
      path: third_party/dinov2/dinov2_vitl14_pretrain.pth
      sha256: d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428
    bank: a23ea9d178ccb05bbe2ee8de5dcab6d31d2fe2b2ea0b5fed2fb939cb9de9b100
    batch: 500
    chosen:
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
    evidence_weights:
      '1':
        B1-QTERM: 2.422641396952869
        B2-KNN/k=100/sign=+: 1.9299495113054856
        B2-KNN/k=1000/sign=+: 2.906375382384673
        B2-KNN/k=100/sign=-: 1.9299495113054856
        B2-KNN/k=1000/sign=-: 2.906375382384673
        B2-KNN/k=10/sign=+: 1.7056188051405254
        B2-KNN/k=10/sign=-: 1.7056188051405254
      '5':
        B1-QTERM: 2.862440950109322
        B2-KNN/k=100/sign=+: 2.2816035414478835
        B2-KNN/k=1000/sign=+: 3.4358020304395187
        B2-KNN/k=100/sign=-: 2.2816035414478835
        B2-KNN/k=1000/sign=-: 3.4358020304395187
        B2-KNN/k=10/sign=+: 2.019013462605367
        B2-KNN/k=10/sign=-: 2.019013462605367
  cub_dinov2_l14:
    dataset: cub
    view: eval84
    weights:
      path: third_party/dinov2/dinov2_vitl14_pretrain.pth
      sha256: d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428
    bank: a2c3d563baf5b5cd01e89f800b0ab6b79db4e259c5ef0659b5f5aca854f68ec7
    batch: 500
    chosen:
      B1 QTERM: B1-QTERM
      B2 KNN: B2-KNN/k=100/sign=-
      B3 RMD: B3-RMD/lambda=1.0
      B4 CL2N: B4-CL2N
      B5 ZN: B5-ZN
      B6 NNN: B6-NNN/alpha=0.75/k=1000
      B7a QB-IS: B7a-QB-IS/beta=10
      B7b QB-DIS: B7b-QB-DIS/beta=10
      B8 TEEN: B8-TEEN
      B9 CSLS-episode (TRANSDUCTIVE REFERENCE): B9-CSLS-episode
    evidence_weights:
      '1':
        B1-QTERM: 3.895514930748064
        B2-KNN/k=100/sign=+: 1.631522981864634
        B2-KNN/k=1000/sign=+: 7.022437937138428
        B2-KNN/k=100/sign=-: 1.631522981864634
        B2-KNN/k=1000/sign=-: 7.022437937138428
        B2-KNN/k=10/sign=+: 1.9253395030683518
        B2-KNN/k=10/sign=-: 1.9253395030683518
      '5':
        B1-QTERM: 4.272016205569243
        B2-KNN/k=100/sign=+: 1.7823081312050717
        B2-KNN/k=1000/sign=+: 7.69073594504281
        B2-KNN/k=100/sign=-: 1.7823081312050717
        B2-KNN/k=1000/sign=-: 7.69073594504281
        B2-KNN/k=10/sign=+: 2.106645982609633
        B2-KNN/k=10/sign=-: 2.106645982609633
  cub_clip_b16_eval224:
    dataset: cub
    view: eval224
    weights:
      path: third_party/clip/ViT-B-16.pt
      sha256: 5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f
    bank: 8d80438fa47fcc86d848ae9c2f645a2b49b2ede5eec18e40fdfe5d910fa10d18
    batch: 500
  cub_dinov2_b14_eval224:
    dataset: cub
    view: eval224
    weights:
      path: third_party/dinov2/dinov2_vitb14_pretrain.pth
      sha256: 0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73
    bank: 2361911de06713763ca00dae392258d8f100d286fb87197bfe368c69b9f3711e
    batch: 500
analysis_cells:
- - cub@minibank_clip_b16
  - mini_clip_b16
  - cub
- - cub@minibank_dinov2_s14
  - mini_dinov2_s14
  - cub
- - cub@minibank_dinov2_b14
  - mini_dinov2_b14
  - cub
- - cub@minibank_dinov2_l14
  - mini_dinov2_l14
  - cub
- - cub_eval224_clip_b16
  - cub_clip_b16_eval224
  - cub
- - cub_eval224_dinov2_b14
  - cub_dinov2_b14_eval224
  - cub
d1_valA_predictions:
  mini|clip_b16|1: 1.0564131092055304
  mini|clip_b16|5: 1.5611273532203263
  tiered|clip_b16|1: 0.5911897096549933
  tiered|clip_b16|5: 0.8859683180329455
  cifarfs|clip_b16|1: 0.556228995882412
  cifarfs|clip_b16|5: 0.956828782057857
  cub|clip_b16|1: 0.8867219262234751
  cub|clip_b16|5: 1.5251977975586093
  mini|dinov2_s14|1: 0.2652296668781468
  mini|dinov2_s14|5: 0.6043748647780934
  tiered|dinov2_s14|1: 0.3137568542711805
  tiered|dinov2_s14|5: 0.3369236393183157
  cifarfs|dinov2_s14|1: 0.3657419003973851
  cifarfs|dinov2_s14|5: 0.6918541193442989
  cub|dinov2_s14|1: 0.7390133837421111
  cub|dinov2_s14|5: 1.0744097521190379
  mini|dinov2_b14|1: 0.11484032895017893
  mini|dinov2_b14|5: 0.4183449046801152
  tiered|dinov2_b14|1: 0.23718867866710824
  tiered|dinov2_b14|5: 0.08836418933678777
  cifarfs|dinov2_b14|1: 0.44692486090342787
  cifarfs|dinov2_b14|5: 0.7113689888372823
  cub|dinov2_b14|1: 0.632246804700697
  cub|dinov2_b14|5: 0.7771313492434179
  mini|dinov2_l14|1: 0.005274335587838426
  mini|dinov2_l14|5: 0.18561096952265924
  tiered|dinov2_l14|1: 0.24851376216832302
  tiered|dinov2_l14|5: 0.030507782561446994
  cifarfs|dinov2_l14|1: 0.6394245241098456
  cifarfs|dinov2_l14|5: 0.578218956821509
  cub|dinov2_l14|1: 0.6352990316970647
  cub|dinov2_l14|5: 0.7100121372525776
hypotheses: H1 (Holm over 32 cells, one-sided Wilcoxon), H2 (Spearman test D1 vs test gain, one-sided permutation p, 100,000 permutations, seed 20261006), H2b (valA D1 above vs test gain), H2-s / H2b-s
  (mean within dataset x shot rho, within-stratum permutations, 100,000 draws, seed 20261007), H3 descriptive
technical_checks: sha256 of every weight, bank and manifest; no NaN
outputs:
  summary: results/foundation_round/test.json, test.md, d1_test.png
  per_episode: results/foundation_round/test_episodes.npz
decisions_after_test: none
```
