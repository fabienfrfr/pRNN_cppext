op_version_set = 0
def forward(self,
    x: Tensor) -> Tensor:
  _0 = ops.prim.NumToTensor(torch.size(x, 0))
  _1 = int(_0)
  _2 = int(_0)
  _3 = ops.prim.NumToTensor(torch.size(x, 1))
  _4 = int(_3)
  input_1 = torch.view(x, [_2, int(_3), 1])
  input_2 = torch._convolution(input_1, getattr(getattr(self.Layers, "13"), "0").weight, getattr(getattr(self.Layers, "13"), "0").bias, [1], [0], [1], False, [0], 64, False, False, True)
  _5 = torch.view(torch.threshold(input_2, 0., 0.), [_1, _4])
  _6 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _7 = torch.select(torch.unsqueeze(CONSTANTS.c1, 1), 2, 0)
  _8 = torch.to(_6, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _9 = torch.index(_7, [_8])
  _10 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _11 = torch.select(torch.unsqueeze(_5, 1), 2, 21)
  _12 = torch.to(_10, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _13 = torch.index(_11, [_12])
  _14 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _15 = torch.select(torch.unsqueeze(CONSTANTS.c2, 1), 2, 0)
  _16 = torch.to(_14, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _17 = torch.index(_15, [_16])
  _18 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _19 = torch.select(torch.unsqueeze(_5, 1), 2, 34)
  _20 = torch.to(_18, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _21 = torch.index(_19, [_20])
  _22 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _23 = torch.select(torch.unsqueeze(_5, 1), 2, 50)
  _24 = torch.to(_22, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _25 = torch.index(_23, [_24])
  _26 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _27 = torch.select(torch.unsqueeze(CONSTANTS.c2, 1), 2, 0)
  _28 = torch.to(_26, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _29 = torch.index(_27, [_28])
  _30 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _31 = torch.select(torch.unsqueeze(_5, 1), 2, 11)
  _32 = torch.to(_30, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _33 = torch.index(_31, [_32])
  _34 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _35 = torch.select(torch.unsqueeze(_5, 1), 2, 13)
  _36 = torch.to(_34, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _37 = torch.index(_35, [_36])
  _38 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _39 = torch.select(torch.unsqueeze(_5, 1), 2, 23)
  _40 = torch.to(_38, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _41 = torch.index(_39, [_40])
  _42 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _43 = torch.select(torch.unsqueeze(_5, 1), 2, 33)
  _44 = torch.to(_42, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _45 = torch.index(_43, [_44])
  _46 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _47 = torch.select(torch.unsqueeze(_5, 1), 2, 29)
  _48 = torch.to(_46, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _49 = torch.index(_47, [_48])
  _50 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _51 = torch.select(torch.unsqueeze(_5, 1), 2, 56)
  _52 = torch.to(_50, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _53 = torch.index(_51, [_52])
  _54 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _55 = torch.select(torch.unsqueeze(_5, 1), 2, 22)
  _56 = torch.to(_54, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _57 = torch.index(_55, [_56])
  _58 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _59 = torch.select(torch.unsqueeze(_5, 1), 2, 63)
  _60 = torch.to(_58, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _61 = torch.index(_59, [_60])
  _62 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _63 = torch.select(torch.unsqueeze(CONSTANTS.c3, 1), 2, 0)
  _64 = torch.to(_62, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _65 = torch.index(_63, [_64])
  _66 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _67 = torch.select(torch.unsqueeze(_5, 1), 2, 5)
  _68 = torch.to(_66, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _69 = torch.index(_67, [_68])
  _70 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _71 = torch.select(torch.unsqueeze(_5, 1), 2, 35)
  _72 = torch.to(_70, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _73 = torch.index(_71, [_72])
  _74 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _75 = torch.select(torch.unsqueeze(_5, 1), 2, 55)
  _76 = torch.to(_74, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _77 = torch.index(_75, [_76])
  _78 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _79 = torch.select(torch.unsqueeze(_5, 1), 2, 3)
  _80 = torch.to(_78, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _81 = torch.index(_79, [_80])
  _82 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _83 = torch.select(torch.unsqueeze(CONSTANTS.c4, 1), 2, 1)
  _84 = torch.to(_82, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _85 = torch.index(_83, [_84])
  _86 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _87 = torch.select(torch.unsqueeze(CONSTANTS.c4, 1), 2, 0)
  _88 = torch.to(_86, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _89 = torch.index(_87, [_88])
  _90 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _91 = torch.select(torch.unsqueeze(_5, 1), 2, 15)
  _92 = torch.to(_90, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _93 = torch.index(_91, [_92])
  _94 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _95 = torch.select(torch.unsqueeze(_5, 1), 2, 2)
  _96 = torch.to(_94, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _97 = torch.index(_95, [_96])
  _98 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _99 = torch.select(torch.unsqueeze(_5, 1), 2, 38)
  _100 = torch.to(_98, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _101 = torch.index(_99, [_100])
  _102 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _103 = torch.select(torch.unsqueeze(_5, 1), 2, 44)
  _104 = torch.to(_102, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _105 = torch.index(_103, [_104])
  _106 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _107 = torch.select(torch.unsqueeze(_5, 1), 2, 36)
  _108 = torch.to(_106, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _109 = torch.index(_107, [_108])
  _110 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _111 = torch.select(torch.unsqueeze(_5, 1), 2, 14)
  _112 = torch.to(_110, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _113 = torch.index(_111, [_112])
  _114 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _115 = torch.select(torch.unsqueeze(_5, 1), 2, 16)
  _116 = torch.to(_114, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _117 = torch.index(_115, [_116])
  _118 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _119 = torch.select(torch.unsqueeze(_5, 1), 2, 60)
  _120 = torch.to(_118, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _121 = torch.index(_119, [_120])
  _122 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _123 = torch.select(torch.unsqueeze(CONSTANTS.c4, 1), 2, 2)
  _124 = torch.to(_122, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _125 = torch.index(_123, [_124])
  _126 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _127 = torch.select(torch.unsqueeze(_5, 1), 2, 18)
  _128 = torch.to(_126, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _129 = torch.index(_127, [_128])
  _130 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _131 = torch.select(torch.unsqueeze(_5, 1), 2, 4)
  _132 = torch.to(_130, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _133 = torch.index(_131, [_132])
  _134 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _135 = torch.select(torch.unsqueeze(_5, 1), 2, 32)
  _136 = torch.to(_134, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _137 = torch.index(_135, [_136])
  _138 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _139 = torch.select(torch.unsqueeze(_5, 1), 2, 49)
  _140 = torch.to(_138, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _141 = torch.index(_139, [_140])
  _142 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _143 = torch.select(torch.unsqueeze(_5, 1), 2, 57)
  _144 = torch.to(_142, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _145 = torch.index(_143, [_144])
  _146 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _147 = torch.select(torch.unsqueeze(_5, 1), 2, 7)
  _148 = torch.to(_146, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _149 = torch.index(_147, [_148])
  _150 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _151 = torch.select(torch.unsqueeze(_5, 1), 2, 24)
  _152 = torch.to(_150, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _153 = torch.index(_151, [_152])
  _154 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _155 = torch.select(torch.unsqueeze(_5, 1), 2, 41)
  _156 = torch.to(_154, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _157 = torch.index(_155, [_156])
  _158 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _159 = torch.select(torch.unsqueeze(_5, 1), 2, 30)
  _160 = torch.to(_158, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _161 = torch.index(_159, [_160])
  _162 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _163 = torch.select(torch.unsqueeze(_5, 1), 2, 26)
  _164 = torch.to(_162, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _165 = torch.index(_163, [_164])
  _166 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _167 = torch.select(torch.unsqueeze(_5, 1), 2, 6)
  _168 = torch.to(_166, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _169 = torch.index(_167, [_168])
  _170 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _171 = torch.select(torch.unsqueeze(CONSTANTS.c5, 1), 2, 0)
  _172 = torch.to(_170, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _173 = torch.index(_171, [_172])
  _174 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _175 = torch.select(torch.unsqueeze(_5, 1), 2, 19)
  _176 = torch.to(_174, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _177 = [_9, _13, _17, _21, _25, _29, _33, _37, _41, _45, _49, _53, _57, _61, _65, _69, _73, _77, _81, _85, _89, _93, _97, _101, _105, _109, _113, _117, _121, _125, _129, _133, _137, _141, _145, _149, _153, _157, _161, _165, _169, _173, torch.index(_175, [_176])]
  input = torch.cat(_177, 1)
  _178 = torch.t(getattr(self.Layers, "0").weight)
  _179 = torch.addmm(getattr(self.Layers, "0").bias, input, _178, beta=1, alpha=1)
  return _179