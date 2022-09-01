op_version_set = 0
def forward(self,
    x: Tensor) -> Tensor:
  _0 = ops.prim.NumToTensor(torch.size(x, 0))
  _1 = int(_0)
  _2 = int(_0)
  _3 = ops.prim.NumToTensor(torch.size(x, 1))
  _4 = int(_3)
  input_1 = torch.view(x, [_2, int(_3), 1])
  input_2 = torch._convolution(input_1, getattr(getattr(self.Layers, "5"), "0").weight, getattr(getattr(self.Layers, "5"), "0").bias, [1], [0], [1], False, [0], 16, False, False, True)
  _5 = torch.view(torch.threshold(input_2, 0., 0.), [_1, _4])
  _6 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _7 = torch.select(torch.unsqueeze(_5, 1), 2, 3)
  _8 = torch.to(_6, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  input_3 = torch.cat([torch.index(_7, [_8])], 1)
  _9 = torch.t(getattr(getattr(self.Layers, "3"), "0").weight)
  input_4 = torch.addmm(getattr(getattr(self.Layers, "3"), "0").bias, input_3, _9, beta=1, alpha=1)
  _10 = torch.threshold(input_4, 0., 0.)
  _11 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _12 = torch.select(torch.unsqueeze(_10, 1), 2, 0)
  _13 = torch.to(_11, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  input_5 = torch.cat([torch.index(_12, [_13])], 1)
  _14 = torch.t(getattr(getattr(self.Layers, "4"), "0").weight)
  input_6 = torch.addmm(getattr(getattr(self.Layers, "4"), "0").bias, input_5, _14, beta=1, alpha=1)
  _15 = torch.threshold(input_6, 0., 0.)
  _16 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _17 = torch.select(torch.unsqueeze(_15, 1), 2, 0)
  _18 = torch.to(_16, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  input_7 = torch.cat([torch.index(_17, [_18])], 1)
  _19 = torch.t(getattr(getattr(self.Layers, "2"), "0").weight)
  input_8 = torch.addmm(getattr(getattr(self.Layers, "2"), "0").bias, input_7, _19, beta=1, alpha=1)
  _20 = torch.threshold(input_8, 0., 0.)
  _21 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _22 = torch.select(torch.unsqueeze(_20, 1), 2, 5)
  _23 = torch.to(_21, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _24 = torch.index(_22, [_23])
  _25 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _26 = torch.select(torch.unsqueeze(_20, 1), 2, 2)
  _27 = torch.to(_25, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  input_9 = torch.cat([_24, torch.index(_26, [_27])], 1)
  _28 = torch.t(getattr(getattr(self.Layers, "1"), "0").weight)
  input_10 = torch.addmm(getattr(getattr(self.Layers, "1"), "0").bias, input_9, _28, beta=1, alpha=1)
  _29 = torch.threshold(input_10, 0., 0.)
  _30 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _31 = torch.select(torch.unsqueeze(_29, 1), 2, 1)
  _32 = torch.to(_30, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _33 = torch.index(_31, [_32])
  _34 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _35 = torch.select(torch.unsqueeze(_29, 1), 2, 4)
  _36 = torch.to(_34, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _37 = torch.index(_35, [_36])
  _38 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _39 = torch.select(torch.unsqueeze(_29, 1), 2, 3)
  _40 = torch.to(_38, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _41 = torch.index(_39, [_40])
  _42 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _43 = torch.select(torch.unsqueeze(_20, 1), 2, 4)
  _44 = torch.to(_42, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _45 = torch.index(_43, [_44])
  _46 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _47 = torch.select(torch.unsqueeze(_5, 1), 2, 12)
  _48 = torch.to(_46, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _49 = torch.index(_47, [_48])
  _50 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _51 = torch.select(torch.unsqueeze(_5, 1), 2, 4)
  _52 = torch.to(_50, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _53 = torch.index(_51, [_52])
  _54 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _55 = torch.select(torch.unsqueeze(_5, 1), 2, 7)
  _56 = torch.to(_54, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _57 = torch.index(_55, [_56])
  _58 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _59 = torch.select(torch.unsqueeze(_29, 1), 2, 0)
  _60 = torch.to(_58, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _61 = torch.index(_59, [_60])
  _62 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _63 = torch.select(torch.unsqueeze(_5, 1), 2, 10)
  _64 = torch.to(_62, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _65 = torch.index(_63, [_64])
  _66 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _67 = torch.select(torch.unsqueeze(_10, 1), 2, 0)
  _68 = torch.to(_66, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _69 = torch.index(_67, [_68])
  _70 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _71 = torch.select(torch.unsqueeze(_5, 1), 2, 6)
  _72 = torch.to(_70, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _73 = torch.index(_71, [_72])
  _74 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _75 = torch.select(torch.unsqueeze(_5, 1), 2, 8)
  _76 = torch.to(_74, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _77 = torch.index(_75, [_76])
  _78 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _79 = torch.select(torch.unsqueeze(_20, 1), 2, 2)
  _80 = torch.to(_78, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _81 = torch.index(_79, [_80])
  _82 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _83 = torch.select(torch.unsqueeze(_5, 1), 2, 3)
  _84 = torch.to(_82, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _85 = torch.index(_83, [_84])
  _86 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _87 = torch.select(torch.unsqueeze(_5, 1), 2, 1)
  _88 = torch.to(_86, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _89 = torch.index(_87, [_88])
  _90 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _91 = torch.select(torch.unsqueeze(_5, 1), 2, 9)
  _92 = torch.to(_90, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _93 = torch.index(_91, [_92])
  _94 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _95 = torch.select(torch.unsqueeze(_5, 1), 2, 13)
  _96 = torch.to(_94, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _97 = torch.index(_95, [_96])
  _98 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _99 = torch.select(torch.unsqueeze(_20, 1), 2, 6)
  _100 = torch.to(_98, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _101 = torch.index(_99, [_100])
  _102 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _103 = torch.select(torch.unsqueeze(_15, 1), 2, 0)
  _104 = torch.to(_102, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _105 = torch.index(_103, [_104])
  _106 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _107 = torch.select(torch.unsqueeze(_20, 1), 2, 3)
  _108 = torch.to(_106, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _109 = torch.index(_107, [_108])
  _110 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _111 = torch.select(torch.unsqueeze(_5, 1), 2, 5)
  _112 = torch.to(_110, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _113 = torch.index(_111, [_112])
  _114 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _115 = torch.select(torch.unsqueeze(_5, 1), 2, 2)
  _116 = torch.to(_114, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _117 = torch.index(_115, [_116])
  _118 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _119 = torch.select(torch.unsqueeze(_20, 1), 2, 7)
  _120 = torch.to(_118, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _121 = torch.index(_119, [_120])
  _122 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _123 = torch.select(torch.unsqueeze(_29, 1), 2, 2)
  _124 = torch.to(_122, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _125 = torch.index(_123, [_124])
  _126 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _127 = torch.select(torch.unsqueeze(_5, 1), 2, 15)
  _128 = torch.to(_126, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _129 = torch.index(_127, [_128])
  _130 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _131 = torch.select(torch.unsqueeze(_20, 1), 2, 0)
  _132 = torch.to(_130, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _133 = torch.index(_131, [_132])
  _134 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _135 = torch.select(torch.unsqueeze(_20, 1), 2, 1)
  _136 = torch.to(_134, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _137 = torch.index(_135, [_136])
  _138 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _139 = torch.select(torch.unsqueeze(_20, 1), 2, 5)
  _140 = torch.to(_138, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _141 = torch.index(_139, [_140])
  _142 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _143 = torch.select(torch.unsqueeze(_5, 1), 2, 11)
  _144 = torch.to(_142, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _145 = torch.index(_143, [_144])
  _146 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _147 = torch.select(torch.unsqueeze(_5, 1), 2, 0)
  _148 = torch.to(_146, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _149 = torch.index(_147, [_148])
  _150 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _151 = torch.select(torch.unsqueeze(_5, 1), 2, 14)
  _152 = torch.to(_150, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _153 = torch.index(_151, [_152])
  _154 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _155 = torch.select(torch.unsqueeze(_29, 1), 2, 3)
  _156 = torch.to(_154, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _157 = torch.index(_155, [_156])
  _158 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _159 = torch.select(torch.unsqueeze(_5, 1), 2, 1)
  _160 = torch.to(_158, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _161 = torch.index(_159, [_160])
  _162 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _163 = torch.select(torch.unsqueeze(_5, 1), 2, 2)
  _164 = torch.to(_162, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _165 = torch.index(_163, [_164])
  _166 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _167 = torch.select(torch.unsqueeze(_5, 1), 2, 8)
  _168 = torch.to(_166, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _169 = torch.index(_167, [_168])
  _170 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _171 = torch.select(torch.unsqueeze(_29, 1), 2, 2)
  _172 = torch.to(_170, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _173 = torch.index(_171, [_172])
  _174 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _175 = torch.select(torch.unsqueeze(_5, 1), 2, 14)
  _176 = torch.to(_174, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _177 = torch.index(_175, [_176])
  _178 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _179 = torch.select(torch.unsqueeze(_5, 1), 2, 5)
  _180 = torch.to(_178, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _181 = torch.index(_179, [_180])
  _182 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _183 = torch.select(torch.unsqueeze(_29, 1), 2, 4)
  _184 = torch.to(_182, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _185 = torch.index(_183, [_184])
  _186 = torch.to(CONSTANTS.c0, torch.device("cpu"), 4, False, False)
  _187 = torch.select(torch.unsqueeze(_5, 1), 2, 8)
  _188 = torch.to(_186, dtype=4, layout=0, device=torch.device("cpu"), non_blocking=False, copy=False)
  _189 = [_33, _37, _41, _45, _49, _53, _57, _61, _65, _69, _73, _77, _81, _85, _89, _93, _97, _101, _105, _109, _113, _117, _121, _125, _129, _133, _137, _141, _145, _149, _153, _157, _161, _165, _169, _173, _177, _181, _185, torch.index(_187, [_188])]
  input = torch.cat(_189, 1)
  _190 = torch.t(getattr(self.Layers, "0").weight)
  _191 = torch.addmm(getattr(self.Layers, "0").bias, input, _190, beta=1, alpha=1)
  return _191
