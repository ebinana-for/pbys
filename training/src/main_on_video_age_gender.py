import argparse
import cv2
import insightface
from src.age_gender.inference import InferenceEngineAgeGender


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
    detector = insightface.model_zoo.get_model('../ressources/det_500m.onnx')
    detector.prepare(ctx_id=0, input_size=(640, 640))

    age_gender_inference = InferenceEngineAgeGender(args.model_path)

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
            try:
                faces = detector.detect(frame)[0]
                for face in faces:
                    xmin=int(face[0])
                    xmax=int(face[2])
                    ymin=int(face[1])
                    ymax=int(face[3])
                    crop = frame[ymin:ymax,xmin:xmax]
                    crop = cv2.resize(crop, (224,224))
                    pred = age_gender_inference.predict([crop])[0]
                    print(pred)
                    gender = pred[0]
                    age = pred[1]
                    frame  = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                    frame = cv2.putText(frame, gender + ' ' + str(age), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            except:
                print("failed inference")
            if args.out_path:
                result.write(frame)
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
