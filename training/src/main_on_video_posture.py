import argparse
import cv2
from ultralytics import YOLO
from src.posture.inference import InferenceEnginePosture

mapping= {"0": 'Too_close_to_airbag', 
          "1": 'Pelvis rotation', 
          "2": 'Torso_dissimetry', 
          "3": 'One_shoulder_in_contact',
          "4": 'Legs_on_dashboard', 
          "5": 'Reaching_back', 
          "6": 'Lateral_body_contact'
          }

def main():
    parser = argparse.ArgumentParser("Age and gender video inference")
    parser.add_argument(
        '-m',"--model_path", required=True, help="Path to input model"
     )
    parser.add_argument(
        '-v',"--video_path", required=False, help="Path to input video"
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
    model = YOLO("yolo11n-pose.pt")

    posture_inference = InferenceEnginePosture(args.model_path)

    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
    elif args.camera_id is not None:
        cap = cv2.VideoCapture(args.camera_id)
    else:
        print("please set video_path or camera_id")

    if args.out_path:

        frame_width = int(cap.get(3)) 
        frame_height = int(cap.get(4)) 
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        size = (frame_width, frame_height) 

        result = cv2.VideoWriter(args.out_path,  
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                fps, size) 

    while(cap.isOpened()):
        ret, frame = cap.read() 
        if ret:
            if True:
                yolo_results = model(frame)
                for yolo_res in yolo_results:
                    box = yolo_res.boxes.xyxy
                    box = box.cpu().numpy()[0]
                    xmin = int(box[0])
                    ymin = int(box[1])
                    xmax = int(box[2])
                    ymax = int(box[3])
                    #draw kpts
                    kpts= yolo_res.keypoints.xy
                    kpts = kpts.cpu().numpy()[0]
                    association = [(6,5), (5,11),(11,12),(12,6), (5,7),(7,9), (6,8),(8,10),(12,14),(11,13)]
                    for couple in association:
                        pt1 = [int(c) for c in kpts[couple[0]]]
                        pt2 = [int(c) for c in kpts[couple[1]]]
                        if pt1 == [0,0] or pt2 == [0,0]:
                            continue
                        frame = cv2.line(frame, pt1, pt2,(255,255,255),10) 
                    crop = frame[ymin:ymax,xmin:xmax]
                    crop = cv2.resize(crop, (224,320))
                    pred = posture_inference.predict([crop])[0]
                    frame  = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),  (0,255,0), 2)
                    for i, p in enumerate(pred):
                        color = (0,255,0)
                        
                        if p < 0.5:
                            color = (0,255,0)
                        else:
                            color = (0,0,255)
                        frame = cv2.putText(frame, str(mapping[str(i)]) + " conf " + str(p), (40, 40+30*i), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            # except:
            #     print("failed inference")
            if args.out_path:
                result.write(frame)
            frame = cv2.resize(frame, (1280,992))
            if args.show:
                cv2.imshow("Image", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
        

    cap.release()
    if args.out_path:
        result.release() 
    if args.show:
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
