pred_label("M") :- ratio(V_0_rr), V_0_rr >= 112, V_0_rr >= 76, V_0_rr <= 244, V_0_rr >= 52, V_0_rr >= 162, V_0_rr <= 162, V_0_rr <= 195.
pred_label("M") :- ratio(V_0_rr), not bentCoiledTail, V_0_rr <= 112, V_0_rr >= 76, V_0_rr <= 76, V_0_rr <= 244, V_0_rr >= 52, V_0_rr <= 162, V_0_rr <= 195.
pred_label("M") :- ratio(V_0_rr), ratioHead(V_0_rh), not bentCoiledTail, V_0_rr >= 195.
pred_label("M") :- ratioHead(V_0_rh), V_0_rh >= 3.
pred_label("M") :- ratio(V_0_rr), not bentNeck, V_0_rr >= 44, V_0_rr <= 112, V_0_rr <= 76, V_0_rr <= 345, V_0_rr <= 244, V_0_rr <= 52, V_0_rr <= 162, V_0_rr <= 195.
pred_label("M") :- ratio(V_0_rr), V_0_rr >= 112, V_0_rr >= 76, V_0_rr <= 345, V_0_rr <= 244, V_0_rr >= 45, V_0_rr >= 244, V_0_rr >= 52, V_0_rr >= 162, V_0_rr >= 195.
pred_label("m") :- bentNeck.
pred_label("m") :- ratio(V_0_rr), V_0_rr >= 71, V_0_rr <= 71.
pred_label("m") :- distalDroplets.
pred_label("m") :- ratio(V_0_rr), V_0_rr <= 79, V_0_rr >= 37, V_0_rr >= 71, V_0_rr >= 79, V_0_rr >= 44.
pred_label("M") :- dagDefect.
pred_label("n") :- ratio(V_0_rr), not bentCoiledTail, V_0_rr >= 112, V_0_rr >= 56, V_0_rr >= 82, V_0_rr >= 53, V_0_rr >= 100, V_0_rr >= 64, V_0_rr >= 42, V_0_rr >= 83, V_0_rr >= 61, V_0_rr >= 164, V_0_rr >= 84, V_0_rr >= 157, V_0_rr <= 185.
pred_label("n") :- ratio(V_0_rr), not detachedHead, V_0_rr <= 112, V_0_rr <= 157, V_0_rr <= 82, V_0_rr >= 53, V_0_rr >= 42, V_0_rr <= 185, V_0_rr <= 100, V_0_rr <= 64, V_0_rr <= 83, V_0_rr <= 164, V_0_rr <= 84, V_0_rr <= 178.
pred_label("n") :- ratio(V_0_rr), V_0_rr >= 112, V_0_rr >= 56, V_0_rr >= 82, V_0_rr >= 53, V_0_rr >= 178, V_0_rr >= 100, V_0_rr >= 64, V_0_rr >= 42, V_0_rr >= 83, V_0_rr >= 236, V_0_rr >= 61, V_0_rr >= 185, V_0_rr >= 164, V_0_rr >= 84, V_0_rr >= 157, V_0_rr <= 236.
pred_label("n") :- ratio(V_0_rr), not dagDefect, not detachedHead, not bentCoiledTail, V_0_rr <= 157, V_0_rr >= 56, V_0_rr >= 82, V_0_rr >= 53, V_0_rr >= 64, V_0_rr >= 42, V_0_rr >= 61, V_0_rr <= 164.
pred_label("n") :- ratio(V_0_rr), not proximalDroplets, not bentCoiledTail, V_0_rr >= 112, V_0_rr >= 56, V_0_rr >= 82, V_0_rr <= 465, V_0_rr >= 53, V_0_rr >= 178, V_0_rr >= 100, V_0_rr >= 64, V_0_rr >= 42, V_0_rr >= 83, V_0_rr >= 236, V_0_rr >= 61, V_0_rr >= 185, V_0_rr >= 267, V_0_rr >= 164, V_0_rr >= 84, V_0_rr >= 157.
pred_label("n") :- ratio(V_0_rr), not dagDefect, not distalDroplets, V_0_rr >= 36, V_0_rr <= 42.
pred_label("m") :- ratio(V_0_rr), V_0_rr >= 31, V_0_rr <= 79, V_0_rr <= 71, V_0_rr <= 37, V_0_rr <= 31, V_0_rr <= 105, V_0_rr <= 98, V_0_rr <= 112, V_0_rr <= 60.
pred_label("m") :- bentCoiledTail.
pred_label("m") :- detachedHead.
pred_label("n") :- headArea(V_0_a), not dagDefect, not bentCoiledTail, V_0_a >= 2450, V_0_a <= 2477.
