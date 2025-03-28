from dumbo_asp.primitives.models import Model
import json
from xasp.entities import Explain

def explain_asp(bg = "bg.lp", asp = "to_expl.lp", show_dag = True):
    with open(bg, 'r') as file:
        program = file.read()
    with open(asp, 'r') as file:
        program = program + file.read()

    explain = Explain.the_program(
        program,
        the_answer_set = Model.of_program(program),
        the_atoms_to_explain = Model.of_program("found_correct_explaination_vit.", "found_correct_explaination_an.")
    )

    # print(explain.explanation_sequence())

    with open("dag.json", "w+") as out:
        out.write(json.dumps(explain.navigator_graph()))

    if show_dag:
        try:
            i = 0
            while True:
                explain.show_navigator_graph(i)
                i += 1
        except:
            pass
    