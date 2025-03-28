
pred_pred_label(C, "m") :- bentNeck(C).
pred_pred_label(C, "m") :- ratio(C, V_0_rr), V_0_rr >= 45, V_0_rr <= 148, V_0_rr <= 112, V_0_rr <= 139, V_0_rr >= 98, V_0_rr >= 77, V_0_rr <= 136.
pred_pred_label(C, "M") :- ratio(C, V_0_rr), V_0_rr >= 135, V_0_rr <= 244, V_0_rr >= 244, V_0_rr >= 81, V_0_rr >= 161, V_0_rr >= 166.
pred_pred_label(C, "M") :- ratio(C, V_0_rr), not bentCoiledTail(C), V_0_rr >= 135, V_0_rr >= 76, V_0_rr <= 135, V_0_rr >= 56, V_0_rr >= 68, V_0_rr >= 46, V_0_rr >= 52, V_0_rr <= 143, V_0_rr >= 41, V_0_rr >= 73, V_0_rr >= 74, V_0_rr >= 65, V_0_rr >= 81, V_0_rr <= 166, V_0_rr <= 161.
pred_pred_label(C, "M") :- proximalDroplets(C).
pred_pred_label(C, "n") :- headArea(C, V_0_a), not dagDefect(C), not detachedHead(C), V_0_a >= 2209, V_0_a <= 2405.
pred_pred_label(C, "M") :- ratioHead(C, V_0_rh), V_0_rh >= 3.
pred_pred_label(C, "M") :- dagDefect(C).
pred_pred_label(C, "m") :- ratio(C, V_0_rr), V_0_rr >= 99, V_0_rr >= 112, V_0_rr >= 139, V_0_rr >= 148, V_0_rr >= 98, V_0_rr >= 135, V_0_rr <= 277, V_0_rr >= 106, V_0_rr >= 136, V_0_rr >= 103, V_0_rr >= 259.
pred_pred_label(C, "m") :- bentCoiledTail(C).
pred_pred_label(C, "m") :- ratio(C, V_0_rr), V_0_rr >= 45, V_0_rr >= 99, V_0_rr >= 112, V_0_rr >= 139, V_0_rr >= 349, V_0_rr >= 148.
pred_pred_label(C, "m") :- distalDroplets(C), ratio(C, V_0_rr), V_0_rr <= 70.
pred_pred_label(C, "M") :- ratio(C, V_0_rr), V_0_rr >= 45, V_0_rr >= 135, V_0_rr >= 76, V_0_rr >= 44, V_0_rr <= 295, V_0_rr >= 56, V_0_rr >= 68, V_0_rr >= 46, V_0_rr >= 52, V_0_rr <= 249, V_0_rr <= 244, V_0_rr <= 143, V_0_rr >= 143.
pred_pred_label(C, "m") :- detachedHead(C).
pred_pred_label(C, "n") :- ratio(C, V_0_rr), not dagDefect(C), not distalDroplets(C), not detachedHead(C), V_0_rr >= 45, V_0_rr >= 44, V_0_rr >= 65, V_0_rr <= 243, V_0_rr >= 137, V_0_rr >= 78, V_0_rr >= 80, V_0_rr >= 132, V_0_rr >= 92.
pred_pred_label(C, "n") :- ratio(C, V_0_rr), not dagDefect(C), V_0_rr >= 45, V_0_rr >= 44, V_0_rr >= 65, V_0_rr <= 132, V_0_rr <= 92, V_0_rr >= 78, V_0_rr <= 170, V_0_rr <= 236, V_0_rr <= 212, V_0_rr <= 183, V_0_rr <= 157, V_0_rr <= 227, V_0_rr <= 225, V_0_rr <= 137, V_0_rr <= 173.
pred_pred_label(C, "n") :- ratio(C, V_0_rr), not proximalDroplets(C), not dagDefect(C), not distalDroplets(C), not bentCoiledTail(C), not bentNeck(C), V_0_rr <= 45, V_0_rr <= 65, V_0_rr >= 34, V_0_rr <= 78, V_0_rr <= 80, V_0_rr <= 132, V_0_rr <= 92, V_0_rr <= 170, V_0_rr <= 236, V_0_rr <= 212, V_0_rr <= 183, V_0_rr <= 157, V_0_rr <= 227, V_0_rr <= 225, V_0_rr <= 137, V_0_rr <= 173.
pred_pred_label(C, "n") :- ratio(C, V_0_rr), not dagDefect(C), V_0_rr >= 45, V_0_rr >= 44, V_0_rr >= 65, V_0_rr <= 65, V_0_rr <= 78, V_0_rr <= 80, V_0_rr <= 132, V_0_rr <= 92, V_0_rr <= 170, V_0_rr <= 236, V_0_rr <= 212, V_0_rr <= 183, V_0_rr <= 157, V_0_rr <= 227, V_0_rr <= 225, V_0_rr <= 137, V_0_rr <= 173.
pred_pred_label(C, "n") :- ratio(C, V_0_rr), not proximalDroplets(C), not distalDroplets(C), V_0_rr >= 289.
pred_pred_label(C, "M") :- ratio(C, V_0_rr), not distalDroplets(C), not bentCoiledTail(C), V_0_rr <= 76, V_0_rr >= 68.
pred_pred_label(C, "M") :- ratio(C, V_0_rr), V_0_rr >= 45, V_0_rr <= 76, V_0_rr <= 135, V_0_rr >= 44, V_0_rr <= 56, V_0_rr >= 46, V_0_rr <= 244, V_0_rr <= 68, V_0_rr <= 143, V_0_rr >= 41, V_0_rr <= 81, V_0_rr <= 166, V_0_rr <= 161, V_0_rr <= 73, V_0_rr <= 74, V_0_rr <= 65.