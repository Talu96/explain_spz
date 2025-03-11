
pred_label("alive") :- red(V_0_r), V_0_r >= 217, V_0_r <= 316.
pred_label("alive") :- red(V_0_r), V_0_r >= 71, V_0_r <= 112.
pred_label("alive") :- red(V_0_r), V_0_r <= -100.
pred_label("alive") :- red(V_0_r), V_0_r >= 1773.
pred_label("dead") :- red(V_0_r), V_0_r >= 117, V_0_r <= 170.
pred_label("dead") :- red(V_0_r), V_0_r <= 69, V_0_r >= -91.
pred_label("dead") :- red(V_0_r), V_0_r <= 1664, V_0_r >= 323.