
pred_pred_label(C, "m") :- overtunedHead(C).
pred_pred_label(C, "m") :- ratio(C, V_0_rr), V_0_rr >= 45, V_0_rr <= 148, V_0_rr <= 112, V_0_rr <= 139, V_0_rr >= 98, V_0_rr >= 77, V_0_rr <= 136.
pred_pred_label(C, "M") :- ratio(C, V_0_rr), V_0_rr >= 135, V_0_rr <= 244, V_0_rr >= 244, V_0_rr >= 81, V_0_rr >= 161, V_0_rr >= 166.
pred_pred_label(C, "M") :- ratio(C, V_0_rr), not singleCoilTail(C), V_0_rr >= 135, V_0_rr >= 76, V_0_rr <= 135, V_0_rr >= 56, V_0_rr >= 68, V_0_rr >= 46, V_0_rr >= 52, V_0_rr <= 143, V_0_rr >= 41, V_0_rr >= 73, V_0_rr >= 74, V_0_rr >= 65, V_0_rr >= 81, V_0_rr <= 166, V_0_rr <= 161.
pred_pred_label(C, "M") :- bubbleHead(C).
pred_pred_label(C, "n") :- headArea(C, V_0_a), not curledTail(C), not missingTail(C), V_0_a >= 2209, V_0_a <= 2405.
pred_pred_label(C, "M") :- ratioHead(C, V_0_rh), V_0_rh >= 3.
pred_pred_label(C, "M") :- curledTail(C).
pred_pred_label(C, "m") :- ratio(C, V_0_rr), V_0_rr >= 99, V_0_rr >= 112, V_0_rr >= 139, V_0_rr >= 148, V_0_rr >= 98, V_0_rr >= 135, V_0_rr <= 277, V_0_rr >= 106, V_0_rr >= 136, V_0_rr >= 103, V_0_rr >= 259.
pred_pred_label(C, "m") :- singleCoilTail(C).
pred_pred_label(C, "m") :- ratio(C, V_0_rr), V_0_rr >= 45, V_0_rr >= 99, V_0_rr >= 112, V_0_rr >= 139, V_0_rr >= 349, V_0_rr >= 148.
pred_pred_label(C, "m") :- bubbleTail(C), ratio(C, V_0_rr), V_0_rr <= 70.
pred_pred_label(C, "M") :- ratio(C, V_0_rr), V_0_rr >= 45, V_0_rr >= 135, V_0_rr >= 76, V_0_rr >= 44, V_0_rr <= 295, V_0_rr >= 56, V_0_rr >= 68, V_0_rr >= 46, V_0_rr >= 52, V_0_rr <= 249, V_0_rr <= 244, V_0_rr <= 143, V_0_rr >= 143.
pred_pred_label(C, "m") :- missingTail(C).
pred_pred_label(C, "n") :- ratio(C, V_0_rr), not curledTail(C), not bubbleTail(C), not missingTail(C), V_0_rr >= 45, V_0_rr >= 44, V_0_rr >= 65, V_0_rr <= 243, V_0_rr >= 137, V_0_rr >= 78, V_0_rr >= 80, V_0_rr >= 132, V_0_rr >= 92.
pred_pred_label(C, "n") :- ratio(C, V_0_rr), not curledTail(C), V_0_rr >= 45, V_0_rr >= 44, V_0_rr >= 65, V_0_rr <= 132, V_0_rr <= 92, V_0_rr >= 78, V_0_rr <= 170, V_0_rr <= 236, V_0_rr <= 212, V_0_rr <= 183, V_0_rr <= 157, V_0_rr <= 227, V_0_rr <= 225, V_0_rr <= 137, V_0_rr <= 173.
pred_pred_label(C, "n") :- ratio(C, V_0_rr), not bubbleHead(C), not curledTail(C), not bubbleTail(C), not singleCoilTail(C), not overtunedHead(C), V_0_rr <= 45, V_0_rr <= 65, V_0_rr >= 34, V_0_rr <= 78, V_0_rr <= 80, V_0_rr <= 132, V_0_rr <= 92, V_0_rr <= 170, V_0_rr <= 236, V_0_rr <= 212, V_0_rr <= 183, V_0_rr <= 157, V_0_rr <= 227, V_0_rr <= 225, V_0_rr <= 137, V_0_rr <= 173.
pred_pred_label(C, "n") :- ratio(C, V_0_rr), not curledTail(C), V_0_rr >= 45, V_0_rr >= 44, V_0_rr >= 65, V_0_rr <= 65, V_0_rr <= 78, V_0_rr <= 80, V_0_rr <= 132, V_0_rr <= 92, V_0_rr <= 170, V_0_rr <= 236, V_0_rr <= 212, V_0_rr <= 183, V_0_rr <= 157, V_0_rr <= 227, V_0_rr <= 225, V_0_rr <= 137, V_0_rr <= 173.
pred_pred_label(C, "n") :- ratio(C, V_0_rr), not bubbleHead(C), not bubbleTail(C), V_0_rr >= 289.
pred_pred_label(C, "M") :- ratio(C, V_0_rr), not bubbleTail(C), not singleCoilTail(C), V_0_rr <= 76, V_0_rr >= 68.
pred_pred_label(C, "M") :- ratio(C, V_0_rr), V_0_rr >= 45, V_0_rr <= 76, V_0_rr <= 135, V_0_rr >= 44, V_0_rr <= 56, V_0_rr >= 46, V_0_rr <= 244, V_0_rr <= 68, V_0_rr <= 143, V_0_rr >= 41, V_0_rr <= 81, V_0_rr <= 166, V_0_rr <= 161, V_0_rr <= 73, V_0_rr <= 74, V_0_rr <= 65.