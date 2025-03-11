:- label("n"), bubbleHead.
:- label("n"), bubbleTail. 
:- label("n"), bubbleHead.
:- label("n"), curledTail.
:- label("n"), missingTail.
:- label("n"), singleCoilTail.
:- label("n"), overtunedHead.

label("n").
red(-76).
area(140160).
ratio(61).
headRoundness(81).
headArea(2335).
lenghtHead(191).
ratioHead(2).

pred_label("M") :- ratio(V_0_rr), V_0_rr >= 45, V_0_rr <= 175, V_0_rr <= 76, V_0_rr <= 64, V_0_rr <= 184, V_0_rr >= 64, V_0_rr >= 44, V_0_rr <= 244, V_0_rr <= 295, V_0_rr <= 249, V_0_rr <= 188.
pred_label("M") :- curledTail.
pred_label("M") :- bubbleHead.
pred_label("M") :- ratio(V_0_rr), V_0_rr <= 184, V_0_rr >= 184.
pred_label("M") :- ratio(V_0_rr), V_0_rr <= 34, V_0_rr >= 32, V_0_rr <= 32.
pred_label("M") :- ratio(V_0_rr), V_0_rr <= 76, V_0_rr >= 64, V_0_rr >= 76.
pred_label("M") :- headArea(V_0_a), V_0_a <= 2588, V_0_a >= 2588.

pred_label("n") :- ratio(V_0_rr), ratioHead(V_0_rh), not bubbleHead, not curledTail, not missingTail, not singleCoilTail, not overtunedHead, V_0_rr >= 33, V_0_rr <= 309.
pred_label("n") :- ratio(V_0_rr), not bubbleHead, not curledTail, not singleCoilTail, not overtunedHead, V_0_rr <= 98, V_0_rr <= 57, V_0_rr <= 55, V_0_rr <= 76, V_0_rr >= 33, V_0_rr <= 132, V_0_rr <= 100, V_0_rr <= 110, V_0_rr <= 225, V_0_rr <= 221, V_0_rr <= 211, V_0_rr <= 84, V_0_rr <= 44, V_0_rr <= 73.
pred_label("n") :- ratio(V_0_rr), not bubbleHead, not curledTail, not bubbleTail, not overtunedHead, V_0_rr >= 39, V_0_rr >= 98, V_0_rr >= 57, V_0_rr >= 55, V_0_rr >= 76, V_0_rr <= 349, V_0_rr >= 100, V_0_rr >= 41, V_0_rr >= 84, V_0_rr >= 132, V_0_rr >= 44, V_0_rr >= 73, V_0_rr >= 211, V_0_rr >= 110.
pred_label("n") :- ratio(V_0_rr), not bubbleHead, not overtunedHead, V_0_rr <= 24.

pred_label("m") :- singleCoilTail.
pred_label("m") :- missingTail.
pred_label("m") :- headArea(V_0_a), V_0_a <= 2665, V_0_a >= 2480, V_0_a >= 2665.
pred_label("m") :- overtunedHead.
pred_label("m") :- bubbleTail.
pred_label("m") :- ratio(V_0_rr), V_0_rr >= 106, V_0_rr <= 106, V_0_rr >= 45, V_0_rr >= 63, V_0_rr >= 85, V_0_rr >= 35, V_0_rr >= 55.
pred_label("m") :- ratio(V_0_rr), V_0_rr >= 31, V_0_rr <= 106, V_0_rr <= 45, V_0_rr <= 31, V_0_rr <= 60, V_0_rr <= 98, V_0_rr <= 103, V_0_rr <= 57, V_0_rr <= 70, V_0_rr <= 301, V_0_rr <= 55, V_0_rr <= 227, V_0_rr <= 85, V_0_rr <= 63, V_0_rr <= 35, V_0_rr <= 135, V_0_rr <= 139, V_0_rr <= 112.

found_correct_explaination :- pred_label(L), label(L).

