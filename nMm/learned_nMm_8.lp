
pred_label("n") :- ratio(V_0_rr), not proximalDroplets, not dagDefect, not bentCoiledTail, V_0_rr >= 110, V_0_rr >= 34, V_0_rr >= 60, V_0_rr >= 35, V_0_rr >= 137, V_0_rr >= 122, V_0_rr >= 101, V_0_rr >= 82, V_0_rr >= 102, V_0_rr <= 212.
pred_label("n") :- ratio(V_0_rr), not dagDefect, not distalDroplets, not detachedHead, not bentCoiledTail, V_0_rr >= 34, V_0_rr <= 122, V_0_rr >= 60, V_0_rr >= 35.
pred_label("n") :- ratio(V_0_rr), V_0_rr <= 110, V_0_rr <= 34, V_0_rr <= 60, V_0_rr <= 122, V_0_rr <= 35, V_0_rr <= 101, V_0_rr <= 82, V_0_rr <= 102, V_0_rr <= 31, V_0_rr <= 24.
pred_label("n") :- ratio(V_0_rr), V_0_rr >= 110, V_0_rr >= 34, V_0_rr >= 31, V_0_rr >= 60, V_0_rr >= 35, V_0_rr >= 137, V_0_rr >= 122, V_0_rr >= 101, V_0_rr >= 82, V_0_rr >= 102, V_0_rr <= 267, V_0_rr >= 23, V_0_rr >= 24, V_0_rr >= 152, V_0_rr >= 173, V_0_rr >= 212, V_0_rr >= 185, V_0_rr >= 267, V_0_rr >= 183, V_0_rr >= 147, V_0_rr >= 210.
pred_label("n") :- ratio(V_0_rr), not dagDefect, V_0_rr <= 110, V_0_rr >= 34, V_0_rr <= 60, V_0_rr <= 122, V_0_rr <= 35, V_0_rr <= 101, V_0_rr <= 82, V_0_rr <= 102.
pred_label("m") :- ratio(V_0_rr), V_0_rr >= 105, V_0_rr >= 220, V_0_rr >= 126, V_0_rr >= 182, V_0_rr >= 240, V_0_rr >= 579, V_0_rr >= 38, V_0_rr >= 39, V_0_rr >= 59, V_0_rr >= 110, V_0_rr >= 44, V_0_rr >= 227, V_0_rr >= 36, V_0_rr >= 37, V_0_rr >= 63, V_0_rr >= 71, V_0_rr >= 77, V_0_rr >= 135, V_0_rr >= 136.
pred_label("M") :- dagDefect.
pred_label("n") :- not proximalDroplets, not dagDefect, not bentCoiledTail, headRoundness(V_0_hr), V_0_hr <= 81.
pred_label("M") :- ratio(V_0_rr), not detachedHead, V_0_rr >= 44, V_0_rr >= 36, V_0_rr <= 135, V_0_rr >= 135, V_0_rr >= 32, V_0_rr >= 46, V_0_rr >= 56, V_0_rr >= 64.
pred_label("M") :- proximalDroplets.
pred_label("M") :- ratio(V_0_rr), V_0_rr >= 44, V_0_rr >= 36, V_0_rr <= 135, V_0_rr <= 188, V_0_rr <= 175, V_0_rr >= 46, V_0_rr >= 56, V_0_rr <= 64, V_0_rr >= 64.
pred_label("M") :- ratio(V_0_rr), not distalDroplets, not bentCoiledTail, V_0_rr >= 36, V_0_rr <= 135, V_0_rr <= 56, V_0_rr >= 32.
pred_label("M") :- ratio(V_0_rr), V_0_rr <= 44, V_0_rr <= 36, V_0_rr <= 135, V_0_rr <= 56, V_0_rr >= 32, V_0_rr <= 46, V_0_rr <= 32.
pred_label("M") :- headRoundness(V_0_hr), V_0_hr <= 83, V_0_hr >= 79, V_0_hr <= 82, V_0_hr >= 82, V_0_hr <= 85.
pred_label("m") :- bentNeck.
pred_label("m") :- detachedHead.
pred_label("m") :- bentCoiledTail.
pred_label("m") :- ratio(V_0_rr), V_0_rr <= 220, V_0_rr >= 38, V_0_rr >= 39, V_0_rr >= 59, V_0_rr >= 44, V_0_rr <= 182, V_0_rr <= 110, V_0_rr <= 105, V_0_rr >= 36, V_0_rr >= 70, V_0_rr >= 37, V_0_rr <= 70, V_0_rr <= 227.
pred_label("m") :- distalDroplets, ratio(V_0_rr), V_0_rr <= 37.
pred_label("M") :- ratio(V_0_rr), V_0_rr >= 244, V_0_rr <= 244.