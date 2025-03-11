
pred_label("alive") :- red(V_0_r), V_0_r <= 1305, V_0_r >= 745.
pred_label("alive") :- red(V_0_r), V_0_r <= 256.
pred_label("dead") :- red(V_0_r), V_0_r >= 286, V_0_r <= 682.
pred_label("dead") :- red(V_0_r), V_0_r >= 1374.