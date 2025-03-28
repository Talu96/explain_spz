
pred_label("M") :- ratio(V_0_rr), not detachedHead, V_0_rr >= 44, V_0_rr <= 135, V_0_rr >= 71, V_0_rr >= 73, V_0_rr >= 56, V_0_rr >= 25, V_0_rr >= 36, V_0_rr <= 162, V_0_rr >= 112, V_0_rr >= 52, V_0_rr <= 249, V_0_rr <= 295, V_0_rr <= 244, V_0_rr >= 37, V_0_rr >= 45, V_0_rr <= 252, V_0_rr <= 166, V_0_rr <= 161, V_0_rr >= 42, V_0_rr >= 94.
pred_label("m") :- ratio(V_0_rr), V_0_rr >= 229, V_0_rr >= 265, V_0_rr >= 44, V_0_rr >= 70, V_0_rr >= 76, V_0_rr >= 106, V_0_rr >= 68, V_0_rr <= 265, V_0_rr >= 39, V_0_rr >= 35.
pred_label("m") :- detachedHead.
pred_label("m") :- distalDroplets, ratio(V_0_rr), V_0_rr >= 243.
pred_label("M") :- proximalDroplets.
pred_label("m") :- ratio(V_0_rr), V_0_rr <= 70, V_0_rr <= 106, V_0_rr >= 70, V_0_rr <= 265, V_0_rr <= 243, V_0_rr >= 35.
pred_label("M") :- headArea(V_0_a), V_0_a <= 2499, V_0_a <= 2364, V_0_a >= 2122, V_0_a <= 2353, V_0_a <= 2122, V_0_a <= 2300.
pred_label("M") :- headArea(V_0_a), V_0_a >= 2353, V_0_a <= 2553, V_0_a >= 2364, V_0_a >= 2499, V_0_a >= 2553.
pred_label("M") :- headArea(V_0_a), V_0_a <= 2499, V_0_a <= 2364, V_0_a >= 2300.
pred_label("m") :- bentNeck.
pred_label("M") :- ratio(V_0_rr), V_0_rr >= 44, V_0_rr >= 135, V_0_rr >= 56, V_0_rr >= 25, V_0_rr >= 36, V_0_rr <= 162, V_0_rr >= 112, V_0_rr >= 162, V_0_rr >= 52, V_0_rr <= 249, V_0_rr <= 244, V_0_rr >= 37, V_0_rr >= 45.
pred_label("n") :- ratio(V_0_rr), not proximalDroplets, not dagDefect, not distalDroplets, not bentCoiledTail, V_0_rr >= 106, V_0_rr >= 135, V_0_rr >= 39, V_0_rr <= 250, V_0_rr >= 33, V_0_rr >= 57, V_0_rr >= 53, V_0_rr >= 47, V_0_rr >= 41, V_0_rr >= 56, V_0_rr >= 80, V_0_rr >= 137, V_0_rr >= 163, V_0_rr >= 156, V_0_rr >= 102.
pred_label("n") :- ratio(V_0_rr), V_0_rr <= 106, V_0_rr <= 39, V_0_rr <= 135, V_0_rr <= 38, V_0_rr <= 57, V_0_rr <= 102, V_0_rr <= 33, V_0_rr <= 137, V_0_rr <= 163, V_0_rr <= 80, V_0_rr <= 53, V_0_rr <= 56, V_0_rr <= 47, V_0_rr >= 32, V_0_rr <= 41.
pred_label("n") :- ratio(V_0_rr), V_0_rr <= 106, V_0_rr <= 135, V_0_rr >= 39, V_0_rr <= 57, V_0_rr <= 102, V_0_rr >= 33, V_0_rr >= 47, V_0_rr <= 137, V_0_rr <= 163, V_0_rr <= 80, V_0_rr <= 53, V_0_rr <= 56, V_0_rr <= 47, V_0_rr >= 41.
pred_label("n") :- ratio(V_0_rr), not bentCoiledTail, V_0_rr <= 106, V_0_rr <= 135, V_0_rr <= 57, V_0_rr <= 102, V_0_rr >= 33, V_0_rr >= 38, V_0_rr <= 137, V_0_rr <= 163, V_0_rr <= 80, V_0_rr <= 53, V_0_rr <= 56, V_0_rr <= 47, V_0_rr <= 41.
pred_label("n") :- ratio(V_0_rr), not proximalDroplets, V_0_rr <= 106, V_0_rr <= 39, V_0_rr <= 135, V_0_rr <= 38, V_0_rr <= 57, V_0_rr <= 102, V_0_rr <= 33, V_0_rr <= 22, V_0_rr <= 137, V_0_rr <= 163, V_0_rr <= 80, V_0_rr <= 53, V_0_rr <= 56, V_0_rr <= 47, V_0_rr <= 41, V_0_rr <= 32.
pred_label("n") :- ratio(V_0_rr), not distalDroplets, V_0_rr >= 106, V_0_rr >= 135, V_0_rr >= 39, V_0_rr >= 301, V_0_rr <= 301, V_0_rr >= 33, V_0_rr >= 38, V_0_rr >= 57, V_0_rr >= 53, V_0_rr >= 47, V_0_rr >= 41, V_0_rr >= 56, V_0_rr >= 80, V_0_rr >= 137, V_0_rr >= 163, V_0_rr >= 156, V_0_rr >= 102, V_0_rr >= 227, V_0_rr >= 250.
pred_label("n") :- ratio(V_0_rr), not distalDroplets, not detachedHead, V_0_rr >= 106, V_0_rr >= 135, V_0_rr >= 39, V_0_rr <= 156, V_0_rr >= 33, V_0_rr >= 38, V_0_rr >= 57, V_0_rr >= 53, V_0_rr >= 47, V_0_rr <= 163, V_0_rr >= 41, V_0_rr >= 56, V_0_rr >= 80, V_0_rr >= 137, V_0_rr >= 102.
pred_label("n") :- ratio(V_0_rr), not dagDefect, not distalDroplets, not detachedHead, not bentCoiledTail, V_0_rr <= 106, V_0_rr <= 135, V_0_rr <= 156, V_0_rr >= 53, V_0_rr >= 47, V_0_rr <= 137, V_0_rr <= 163.
pred_label("M") :- ratio(V_0_rr), V_0_rr <= 143, V_0_rr >= 143.
pred_label("M") :- dagDefect, not bentCoiledTail.
pred_label("m") :- ratio(V_0_rr), V_0_rr <= 148, V_0_rr <= 31, V_0_rr <= 106, V_0_rr <= 127, V_0_rr <= 122, V_0_rr <= 77, V_0_rr <= 98, V_0_rr <= 71, V_0_rr <= 135, V_0_rr <= 35, V_0_rr >= 31.
pred_label("m") :- bentCoiledTail.
pred_label("M") :- ratio(V_0_rr), V_0_rr >= 44, V_0_rr <= 71, V_0_rr <= 135, V_0_rr <= 73, V_0_rr <= 56, V_0_rr >= 36, V_0_rr <= 162, V_0_rr <= 143, V_0_rr >= 52, V_0_rr <= 112, V_0_rr <= 52, V_0_rr <= 94, V_0_rr <= 252, V_0_rr <= 166, V_0_rr <= 161, V_0_rr >= 42.

