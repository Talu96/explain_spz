
pred_label("alive") :- red(V_0_r), V_0_r <= 512.
pred_label("dead") :- red(V_0_r), V_0_r >= 525.