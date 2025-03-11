
pred_label("dead") :- red(V_0_r), V_0_r <= 527, V_0_r >= 220.
pred_label("dead") :- red(V_0_r), V_0_r >= 672.
pred_label("alive") :- red(V_0_r), V_0_r <= 665, V_0_r >= 546.
pred_label("alive") :- red(V_0_r), V_0_r <= 204.