
pred_label("dead") :- red(V_0_r), V_0_r <= 220, V_0_r >= 112.
pred_label("dead") :- red(V_0_r), V_0_r >= 498.
pred_label("alive") :- red(V_0_r), V_0_r >= 222, V_0_r <= 463.
pred_label("alive") :- red(V_0_r), V_0_r <= 107.