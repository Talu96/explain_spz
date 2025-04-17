import utils, utils_GUI
from BoxSelector import BoxSelector

def main():
    app = BoxSelector()
    green_rect, label_to_expl, img_name = app.run()
    subimage, features = utils.get_features(img_name, green_rect, label_to_expl)
    
    utils.generate_asp(features)
    utils_GUI.generate_expl(subimage)
    
if __name__ == "__main__":
    main()