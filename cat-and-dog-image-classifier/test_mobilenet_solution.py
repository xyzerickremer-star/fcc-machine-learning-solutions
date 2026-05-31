import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

PATH = 'cats_and_dogs'
test_dir = os.path.join(PATH, 'test')
IMG_HEIGHT = 224
IMG_WIDTH = 224

model = MobileNetV2(weights='imagenet')
cat_labels = {'tabby', 'tiger_cat', 'Persian_cat', 'Siamese_cat', 'Egyptian_cat'}

def dog_probability(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    preds = model.predict(arr, verbose=0)
    decoded = decode_predictions(preds, top=5)[0]
    dog_score = 0.0
    cat_score = 0.0
    for _, label, prob in decoded:
        if label in cat_labels or 'cat' in label.lower():
            cat_score += float(prob)
        if 'dog' in label.lower() or label in {
            'Chihuahua','Japanese_spaniel','Maltese_dog','Pekinese','Shih-Tzu','Blenheim_spaniel','papillon','toy_terrier','Rhodesian_ridgeback','Afghan_hound','basset','beagle','bloodhound','bluetick','black-and-tan_coonhound','Walker_hound','English_foxhound','redbone','borzoi','Irish_wolfhound','Italian_greyhound','whippet','Ibizan_hound','Norwegian_elkhound','otterhound','Saluki','Scottish_deerhound','Weimaraner','Staffordshire_bullterrier','American_Staffordshire_terrier','Bedlington_terrier','Border_terrier','Kerry_blue_terrier','Irish_terrier','Norfolk_terrier','Norwich_terrier','Yorkshire_terrier','wire-haired_fox_terrier','Lakeland_terrier','Sealyham_terrier','Airedale','cairn','Australian_terrier','Dandie_Dinmont','Boston_bull','miniature_schnauzer','giant_schnauzer','standard_schnauzer','Scotch_terrier','Tibetan_terrier','silky_terrier','soft-coated_wheaten_terrier','West_Highland_white_terrier','Lhasa','flat-coated_retriever','curly-coated_retriever','golden_retriever','Labrador_retriever','Chesapeake_Bay_retriever','German_short-haired_pointer','vizsla','English_setter','Irish_setter','Gordon_setter','Brittany_spaniel','clumber','English_springer','Welsh_springer_spaniel','cocker_spaniel','Sussex_spaniel','Irish_water_spaniel','kuvasz','schipperke','groenendael','malinois','briard','kelpie','komondor','Old_English_sheepdog','Shetland_sheepdog','collie','Border_collie','Bouvier_des_Flandres','Rottweiler','German_shepherd','Doberman','miniature_pinscher','Greater_Swiss_Mountain_dog','Bernese_mountain_dog','Appenzeller','EntleBucher','boxer','bull_mastiff','Tibetan_mastiff','French_bulldog','Great_Dane','Saint_Bernard','Eskimo_dog','malamute','Siberian_husky','dalmatian','affenpinscher','basenji','pug','Leonberg','Newfoundland','Great_Pyrenees','Samoyed','Pomeranian','chow','keeshond','Brabancon_griffon','Pembroke','Cardigan','toy_poodle','miniature_poodle','standard_poodle','Mexican_hairless','dingo','dhole','African_hunting_dog'
        }:
            dog_score += float(prob)
    if dog_score + cat_score > 0:
        return dog_score / (dog_score + cat_score)
    top_label = decoded[0][1]
    return 1.0 if 'dog' in top_label.lower() else 0.0

files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
probabilities = [dog_probability(os.path.join(test_dir, f)) for f in files]
answers =  [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
            1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
            1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
            1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
            0, 0, 0, 0, 0, 0]
correct = sum(round(p) == a for p, a in zip(probabilities, answers))
print(correct, len(answers), round(correct / len(answers) * 100, 2))
print([round(p,3) for p in probabilities])
