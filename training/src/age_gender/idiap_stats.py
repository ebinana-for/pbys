import argparse
import cv2
import insightface
from src.age_gender.inference import InferenceEngineAgeGender
import glob
import os
import numpy as np
import random
import tqdm
import pandas as pd
random.seed(10)
import plotly.express as px
from src.age_gender.dataset.data_utils import cat2age, age2cat, fairface_age_categories

def main():
    parser = argparse.ArgumentParser("Age and gender video inference")
    parser.add_argument(
        '-m',"--model_path", required=True, help="Path to input model"
     )
    parser.add_argument(
        '-v',"--videos_path", required=False, help="Path to input video"
    )
    parser.add_argument(
        '-c',"--camera_id", required=False, help="camera id, local only", type=int
    )
    parser.add_argument(
        '-s',"--show",  action='store_true', help="show stream, local only"
    )
    parser.add_argument(
        '-o',"--out_path", required=False, help="Path to store output video"
    )
    args = parser.parse_args()
    detector = insightface.model_zoo.get_model('../ressources/det_500m.onnx')
    detector.prepare(ctx_id=0, input_size=(640, 640), providers=['CUDAExecutionProvider'])

    age_gender_inference = InferenceEngineAgeGender(args.model_path)

    df= pd.DataFrame()
    for vid in tqdm.tqdm(glob.glob(os.path.join(args.videos_path,"*"))):
        age = os.path.basename(vid).split('_')[-1].strip('.mp4')
        cap = cv2.VideoCapture(vid)

        if args.out_path:

            frame_width = int(cap.get(3)) 
            frame_height = int(cap.get(4)) 
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            size = (frame_width, frame_height) 

            result = cv2.VideoWriter(args.out_path,  
                                    cv2.VideoWriter_fourcc(*'mp4v'), 
                                    fps, size) 
        age_list = []
        while(cap.isOpened()):
            ret, frame = cap.read() 
            if ret:
                faces = detector.detect(frame)[0]
                if len(faces) > 1:
                    continue
                for face in faces:
                    xmin=int(face[0])
                    xmax=int(face[2])
                    ymin=int(face[1])
                    ymax=int(face[3])
                    crop = frame[ymin:ymax,xmin:xmax]
                    try:
                        crop = cv2.resize(crop, (224,224))
                        pred = age_gender_inference.predict([crop])[0]
                    except:
                        continue
                    # print(pred)
                    gender = pred[0]
                    age = pred[1]
                    age_list.append(age)
                    frame  = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                    frame = cv2.putText(frame, gender + ' ' + str(age), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                
                if args.out_path:
                    result.write(frame)
                if args.show:
                    cv2.imshow("Image", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                break
        if len(age_list)<30:
            continue
        age_mean = np.mean(age_list)
        age_median = np.median(age_list)
        age_std = np.std(age_list)
        age_random = random.choice(age_list)
        age_diff_median = abs(age_median - age)
        age_diff_mean = abs(age_mean - age)
        age_diff_random = abs(age_random-age)
        age_cat_gt = age2cat(age, fairface_age_categories)
        df_row = pd.DataFrame({ 
                        "age_gt": age, 
                        "age_cat_gt": age_cat_gt, 
                        "age_std": age_std, 
                        "age_mean": age_mean,
                        "age_median": age_median,
                        "age_random": age_random,
                        "age_diff_mean": age_diff_mean,
                        "age_diff_median": age_diff_median,
                        "age_diff_random": age_diff_random,
                        "counter": 1},
                        index=[0]
                        )
        print(df_row)
        df = pd.concat([df, df_row], ignore_index=True)
        cap.release()
        if args.out_path:
            result.release() 
        if args.show:
            cv2.destroyAllWindows()

    print("MEDIAN MAE")
    print(df.age_diff_median.mean())
    print("MEAN MAE")
    print(df.age_diff_mean.mean())
    print("RANDOM CHOICE MAE")
    print(df.age_diff_random.mean())

    # age mean MAE per age category
    fig = px.histogram(df, 
                        x="age_cat_gt", 
                        y="age_diff_mean", 
                        histfunc='avg', 
                        title=f"Age mean MAE per age category for Idiap dataset"
                        )
    fig.update_xaxes(categoryorder='array', categoryarray= fairface_age_categories)
    fig.update_layout(bargap=0.2)
    with open("age_mean_MAE_per_age_category.html", "w") as f:
        f.write(fig.to_html())


    # age median MAE per age category
    fig = px.histogram(df, 
                        x="age_cat_gt", 
                        y="age_diff_median", 
                        histfunc='avg', 
                        title=f"Age median MAE per age category for Idiap dataset"
                        )
    fig.update_xaxes(categoryorder='array', categoryarray= fairface_age_categories)
    fig.update_layout(bargap=0.2)
    with open("age_median_MAE_per_age_category.html", "w") as f:
        f.write(fig.to_html())

    # age random MAE per age category
    fig = px.histogram(df, 
                        x="age_cat_gt", 
                        y="age_diff_random", 
                        histfunc='avg', 
                        title=f"Age random MAE per age category for Idiap dataset"
                        )
    fig.update_xaxes(categoryorder='array', categoryarray= fairface_age_categories)
    fig.update_layout(bargap=0.2)
    with open("age_random_MAE_per_age_category.html", "w") as f:
        f.write(fig.to_html())

    # nb per age category
    fig = px.histogram(df, 
                        x="age_cat_gt", 
                        y="counter", 
                        histfunc='sum',
                        title=f"Nb samples per age category for Idiap dataset"
                        )
    fig.update_xaxes(categoryorder='array', categoryarray= fairface_age_categories)
    fig.update_layout(bargap=0.2)
    with open("nb_samples_per_age_category.html", "w") as f:
        f.write(fig.to_html())
        
    fig = px.histogram(df, 
                    x="age_std", 
                    y="counter", 
                    histfunc='sum',
                    nbins=20,
                    title=f"histo for std"
                    )
    fig.update_layout(bargap=0.2)
    with open("histo_std.html", "w") as f:
        f.write(fig.to_html())

if __name__ == "__main__":
    main()
