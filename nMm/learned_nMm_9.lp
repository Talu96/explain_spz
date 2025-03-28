
pred_label("m") :- ratio(V_0_rr), V_0_rr >= 301, V_0_rr <= 301, V_0_rr >= 227, V_0_rr >= 122, V_0_rr >= 139, V_0_rr >= 147, V_0_rr >= 124, V_0_rr >= 126, V_0_rr >= 148.
pred_label("m") :- bentNeck.
pred_label("M") :- ratio(V_0_rr), not distalDroplets, not bentCoiledTail, V_0_rr <= 337, V_0_rr >= 195, V_0_rr >= 229, V_0_rr >= 76, V_0_rr >= 112, V_0_rr >= 175.
pred_label("M") :- ratio(V_0_rr), not bentCoiledTail, V_0_rr >= 64, V_0_rr <= 143, V_0_rr <= 76, V_0_rr <= 112, V_0_rr >= 76, V_0_rr <= 229, V_0_rr >= 44, V_0_rr <= 175, V_0_rr <= 166, V_0_rr <= 153, V_0_rr <= 161, V_0_rr <= 81, V_0_rr <= 98, V_0_rr >= 41.
pred_label("m") :- ratio(V_0_rr), V_0_rr <= 80, V_0_rr <= 40, V_0_rr >= 40, V_0_rr <= 79, V_0_rr <= 77.
pred_label("M") :- ratio(V_0_rr), not bentNeck, V_0_rr <= 143, V_0_rr <= 64, V_0_rr <= 76, V_0_rr >= 32, V_0_rr <= 195, V_0_rr <= 112, V_0_rr <= 36, V_0_rr <= 229, V_0_rr <= 166, V_0_rr <= 153, V_0_rr <= 161, V_0_rr <= 238, V_0_rr <= 81, V_0_rr <= 41, V_0_rr <= 98.
pred_label("m") :- detachedHead.
pred_label("M") :- headArea(V_0_a), V_0_a >= 2353, V_0_a <= 2396.
pred_label("m") :- headArea(V_0_a), V_0_a >= 2288, V_0_a <= 2288.
pred_label("M") :- distalDroplets, ratio(V_0_rr), V_0_rr <= 143.
pred_label("n") :- ratio(V_0_rr), not bentCoiledTail, V_0_rr <= 64, V_0_rr <= 112, V_0_rr <= 36, V_0_rr <= 37, V_0_rr <= 175, V_0_rr <= 80, V_0_rr <= 79, V_0_rr <= 106, V_0_rr <= 73, V_0_rr <= 102, V_0_rr <= 183, V_0_rr <= 178, V_0_rr <= 157, V_0_rr <= 171, V_0_rr <= 83, V_0_rr <= 55, V_0_rr <= 56, V_0_rr <= 67, V_0_rr <= 30.
pred_label("n") :- ratio(V_0_rr), not distalDroplets, not bentCoiledTail, V_0_rr <= 112, V_0_rr >= 37, V_0_rr <= 175, V_0_rr <= 80, V_0_rr <= 79, V_0_rr >= 55, V_0_rr <= 106, V_0_rr <= 73, V_0_rr <= 102, V_0_rr <= 183, V_0_rr <= 178, V_0_rr <= 157, V_0_rr <= 171, V_0_rr <= 83.
pred_label("n") :- ratio(V_0_rr), not proximalDroplets, V_0_rr <= 64, V_0_rr <= 112, V_0_rr <= 37, V_0_rr >= 37, V_0_rr <= 175, V_0_rr <= 80, V_0_rr <= 79, V_0_rr <= 106, V_0_rr <= 73, V_0_rr <= 102, V_0_rr <= 183, V_0_rr <= 178, V_0_rr <= 157, V_0_rr <= 171, V_0_rr <= 83, V_0_rr <= 55, V_0_rr <= 56, V_0_rr <= 67.
pred_label("n") :- ratio(V_0_rr), V_0_rr >= 175, V_0_rr >= 157, V_0_rr >= 178, V_0_rr >= 465, V_0_rr >= 171, V_0_rr >= 183.
pred_label("M") :- dagDefect.
pred_label("M") :- proximalDroplets.
pred_label("m") :- bentCoiledTail.
pred_label("n") :- ratio(V_0_rr), not proximalDroplets, not dagDefect, not distalDroplets, not detachedHead, not bentCoiledTail, V_0_rr >= 79, V_0_rr <= 183.