pred_label("dead") :- red(V_0_r), V_0_r <= 305, V_0_r >= 218.
pred_label("dead") :- red(V_0_r), V_0_r >= 538.
pred_label("alive") :- red(V_0_r), V_0_r <= 525, V_0_r >= 323.
pred_label("alive") :- red(V_0_r), V_0_r <= 231.