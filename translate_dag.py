import json, re

def correct_explanation(data, type):
    """
    Controlla se esiste un nodo con label "found_correct_explaination".
    
    Param 
    - data: Dizionario contenente i dati JSON
    Return:
    - True se il nodo esiste, None altrimenti
    """
    for node in data.get("nodes", []):
        if "found_correct_explaination_" + type in node.get("label", ""):
            return node.get("id")
    return None

def find_links_with_source(data, source_id):
    """
    Trova tutti i link che hanno come source un determinato id.
    
    Param 
    - data: Dizionario contenente i dati JSON
    - source_id: ID del nodo sorgente
    Return
    - Lista di link con il nodo sorgente specificato
    """
    return [link for link in data.get("links", []) if link.get("source") == source_id]

def find_pred_label_nodes(data, links):
    """
    Trova tutti i nodi target dei link dati che hanno una label che inizia con "pred_label".
    
    Param 
    - data: Dizionario contenente i dati JSON
    - links: Lista di link da analizzare
    Return
    - lista di nodi che hanno una label "pred_label"
    """
    target_ids = {link["target"] for link in links}
    return [node for node in data.get("nodes", []) if node["id"] in target_ids and node["label"].startswith("pred_label")] 

def group_targets_by_link_label(data, nodes):
    """
    Per ogni nodo in nodes, trova tutti i target raggruppati per label del link.
    
    Param 
    - data: Dizionario contenente i dati JSON
    - nodes: Lista di nodi sorgente
    Return
    - Dizionario con chiavi le label dei link e valori le liste di label dei nodi target
    """
    grouped_targets = {}
    for node in nodes:
        node_id = node["id"]
        links = find_links_with_source(data, node_id)
        for link in links:
            label = link.get("label", "unknown")
            target_id = link["target"]
            target_node = next((n for n in data["nodes"] if n["id"] == target_id), None)
            if target_node:
                label = label.split("\n")[0]
                if label not in grouped_targets:
                    grouped_targets[label] = []
                grouped_targets[label].append(target_node["label"].split("\n")[0])
    return grouped_targets

def parse_rule(rule):
    """
    Estrae la classe predetta, i predicati con vincoli e i predicati negati da una stringa di regola logica.
    
    Param
    - rule: Stringa della regola logica (es. 'pred_label("m") :- headArea(V_0_a), not detachedHead, V_0_a <= 2665, V_0_a >= 2480.')
    Return 
    - Tuple (classe_predetta, predicati_con_vincoli, predicati_negati)
    """
    # Trova la classe predetta
    class_match = re.search(r'pred_label\(\"(.*?)\"\)', rule)
    class_pred = class_match.group(1) if class_match else None

    # Trova i predicati senza argomenti (es. detachedHead)
    simple_predicates = re.findall(r'([a-zA-Z]+[,\.])', rule)
    simple_predicates = [s[:-1] for s in simple_predicates]
    
    # Trova i predicati (es. headArea(V_0_a))
    predicates = re.findall(r'(\w+\(V_[0-9]_[a-zA-Z0-9_]+\))', rule)

    # Trova le condizioni sui vincoli (es. V_0_a <= 2665)
    constraints = re.findall(r'(V_[0-9]_[a-zA-Z0-9_]+\s*[<>=]+\s*[\d.]+)', rule)

    # Trova i predicati negati (es. not detachedHead)
    negated_predicates = re.findall(r'not (\w+)', rule)

    # Crea un dizionario per associare variabili a vincoli
    variable_constraints = {}
    for constraint in constraints:
        var_match = re.search(r'(V_[0-9]_[a-zA-Z0-9_]+)', constraint)
        if var_match:
            var = var_match.group(1)
            if var not in variable_constraints:
                variable_constraints[var] = []
            variable_constraints[var].append(constraint)
    
    # Associa i vincoli ai predicati
    predicates_with_constraints = []
    for pred in predicates:
        var_match = re.search(r'\(V_[0-9]_[a-zA-Z0-9_]+\)', pred)
        if var_match:
            var = var_match.group(0)[1:-1]  # Estrae la variabile, es. 'V_0_a'
            constraints_for_pred = variable_constraints.get(var, [])
            predicates_with_constraints.append([pred, constraints_for_pred])

    return class_pred, simple_predicates, predicates_with_constraints, negated_predicates

def parse(nodes):
    keys = nodes.keys()
    rules_parsed = []
    for rule in keys:
        rules_parsed.append(parse_rule(rule))
    return rules_parsed

def explain(p, labels):
    class_pred, simple_predicates, predicates_with_constraints, negated_predicates = p

    class_map = {"n": "normal", "M": "major", "m": "minor"}
    class_pred = class_map.get(class_pred, class_pred)

    expl = f"The {'spermatozoa is' if class_pred not in class_map.values() else 'predicted class is'} {class_pred} because:\n"
    
    constraint_map = {
        "red": "- the spermatozoa is {red}\n",
        "ratio": "- the ratio of the spermatozoa is {ratio} which is in the range: {range}\n",
        "ratioHead": "- the ratio of the head of the spermatozoa is {ratioHead} which is in the range: {range}\n",
        "headRoundness": "- the roundness of the head of the spermatozoa is {headRoundness} which is in the range: {range}\n",
        "headArea": "- the area of the head of the spermatozoa is {headArea} which is in the range: {range}\n",
        "lenghtHead": "- the perimeter of the head of the spermatozoa is {lenghtHead} which is in the range: {range}\n"
    }
    
    predicate_map = {
        "proximalDroplets": "- there is a proximal droplets\n",
        "dagDefect": "- there is a dag defect\n",
        "distalDroplets": "- there is a distal droplets\n",
        "detachedHead": "- it is a detached head\n",
        "bentCoiledTail": "- the tail is bent or coiled\n",
        "bentNeck": "- the neck is bent\n"
    }

    for pr in predicates_with_constraints:
        key = pr[0].split("(")[0]
        if pr[1]:
            a = constraint_map.get(key, "")
            if "{red}" in a:
                if class_pred == "dead":
                    a = a.replace("{red}", "red")
                else:
                    a = a.replace("{red}", "white")
                expl += a
            else:
                expl += a.format(**labels, range=pr[1])
        else:
            expl += predicate_map.get(key, "")

    for s in simple_predicates:
        if s not in negated_predicates:
            expl += predicate_map.get(s, "")

    for neg in negated_predicates:
        if neg in predicate_map:
            expl +=  predicate_map[neg].split("is")[0] + "is not" + predicate_map[neg].split("is")[1]


    return expl

def get_label_from_id(data, id):
    """
    Controlla se esiste un nodo con label "found_correct_explaination".
    
    Param 
    - data: Dizionario contenente i dati JSON
    Return
    - True se il nodo esiste, None altrimenti
    """
    res = {}
    for node in data.get("nodes", []):
        if node.get("id") == id:
            label = node.get("label").split("\n")[0]
            res[label.split("(")[0]] = label.split("(")[1].split(")")[0] if "(" in label else None
    return res

def get_explanation(js, type):
    with open(js) as f:
        dag_json = json.load(f)

    id_correct = correct_explanation(dag_json, type)
    if id_correct is None:
        if type == "an":
            return "Sorry, there is no explanation for the anomaly predicted."
        else:
            return "Sorry, there is no explanation for the vitality predicted."
    
    linked = find_links_with_source(dag_json, id_correct)
    correct_pred_label = find_pred_label_nodes(dag_json, linked) 

    if not correct_pred_label:
        if type == "an":
            return "Sorry, there is no explanation for the anomaly predicted."
        else:
            return "Sorry, there is no explanation for the vitality predicted."
    
    eheh = correct_pred_label[0]["id"]
    linked = find_links_with_source(dag_json, eheh)
    
    labels = {key: value for i in linked for key, value in get_label_from_id(dag_json, i["target"]).items()}

    targets = group_targets_by_link_label(dag_json, correct_pred_label)
    parsed = parse(targets)   
    
    return "\n".join(explain(p, labels) for p in parsed)

def get_expl_from_dag(js = "dag.json"):
    expl_an = get_explanation(js, "an")
    expl_vit = get_explanation(js, "vit")
    return expl_an, expl_vit
