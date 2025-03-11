
pred_label("alive") :- red(V_0_r), V_0_r >= 249, V_0_r <= 275.
pred_label("alive") :- red(V_0_r), V_0_r <= 81.
pred_label("dead") :- red(V_0_r), V_0_r >= 112, V_0_r <= 247.
pred_label("dead") :- red(V_0_r), V_0_r >= 294.