
pred_label("dead") :- red(V_0_r), V_0_r <= 702, V_0_r >= 142.
pred_label("dead") :- red(V_0_r), V_0_r >= 1126.
pred_label("alive") :- red(V_0_r), V_0_r <= 894, V_0_r >= 741.
pred_label("alive") :- red(V_0_r), V_0_r <= 112.