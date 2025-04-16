import sys
import utils
import utils_GUI

def main(img_name):
    global green_rect
    global label_to_expl
    img_name = img_name.split(".")[0]
    green_rect, label_to_expl = utils_GUI.get_rect_to_expl(img_name)
    subimage, features = utils.get_features(img_name, green_rect, label_to_expl)
    utils.generate_asp(features)
    utils.generate_expl(subimage)
    
if __name__ == "__main__":
    img = sys.argv[1]
    main(img)