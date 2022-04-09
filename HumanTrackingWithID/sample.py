
import copy
import time
import argparse

import cv2
import numpy as np
from pupil_apriltags import Detector

from yolox.yolox_onnx import YoloxONNX
from bytetrack.mc_bytetrack import MultiClassByteTrack

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    
    
    parser.add_argument("--families", type=str, default='tag36h11')
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--quad_decimate", type=float, default=2.0)
    parser.add_argument("--quad_sigma", type=float, default=0.0)
    parser.add_argument("--refine_edges", type=int, default=1)
    parser.add_argument("--decode_sharpening", type=float, default=0.25)
    parser.add_argument("--debug", type=int, default=0)

    # YOLOX parameters
    parser.add_argument(
        "--yolox_model",
        type=str,
        default='model/yolox_nano.onnx',
    )
    parser.add_argument(
        '--input_shape',
        type=str,
        default="416,416",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.3,
        help='Class confidence',
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.45,
        help='NMS IoU threshold',
    )
    parser.add_argument(
        '--nms_score_th',
        type=float,
        default=0.1,
        help='NMS Score threshold',
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )

    # motpy parameters
    parser.add_argument(
        "--track_thresh",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--track_buffer",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--min_box_area",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--mot20",
        action="store_true",
    )

    args = parser.parse_args()

    return args


class dict_dot_notation(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def main():

    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.movie is not None:
        cap_device = args.movie
        
    families = args.families
    nthreads = args.nthreads
    quad_decimate = args.quad_decimate
    quad_sigma = args.quad_sigma
    refine_edges = args.refine_edges
    decode_sharpening = args.decode_sharpening
    debug = args.debug

    # YOLOX parameters
    model_path = args.yolox_model
    input_shape = tuple(map(int, args.input_shape.split(',')))
    score_th = args.score_th
    nms_th = args.nms_th
    nms_score_th = args.nms_score_th
    with_p6 = args.with_p6

    # ByteTrack parameters
    track_thresh = args.track_thresh
    track_buffer = args.track_buffer
    match_thresh = args.match_thresh
    min_box_area = args.min_box_area
    mot20 = args.mot20

   
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    
    yolox = YoloxONNX(
        model_path=model_path,
        input_shape=input_shape,
        class_score_th=score_th,
        nms_th=nms_th,
        nms_score_th=nms_score_th,
        with_p6=with_p6,
        providers=['CPUExecutionProvider'],
    )

    
    tracker = MultiClassByteTrack(
        fps=cap_fps,
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        min_box_area=min_box_area,
        mot20=mot20,
    )
    
    at_detector = Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
        debug=debug,
    )
    elapsed_time = 0


   
    track_id_dict = {}
   
    with open('coco_classes.txt', 'rt') as f:
        coco_classes = f.read().rstrip('\n').split('\n')

    while True:
        start_time = time.time()

   
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)
        
        
        ret2, image2 = cap.read()
        if not ret2:
            break
        debug_image2 = copy.deepcopy(image2)


        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        tags = at_detector.detect(
            image2,
            estimate_tag_pose=False,
            camera_params=None,
            tag_size=None,
        )
        
        debug_image2 = draw_tags(debug_image2, tags, elapsed_time)

        elapsed_time = time.time() - start_time



        # Object Detection
        bboxes, scores, class_ids = yolox.inference(frame)


        t_ids, t_bboxes, t_scores, t_class_ids = tracker(
            frame,
            bboxes,
            scores,
            class_ids,
        )
    
        
        """
        Kendi listemi oluşturdum. Manuel olarak ekleceğiz buraya ID leri çünkü sonrasında
        Detect edilen ile eşitlemek için
        """
        manuelUniqList = [1,2,3,4,5,6,7,8,9,10]
                      
        uniq_List = []        
        for x in tags:        
           unique = x.tag_id
           if unique not in uniq_List:                    
              uniq_List.append(unique)
            
        print(uniq_List)    
            
            
        manuelUniqList2 = set(manuelUniqList)
        uniq_List2 = set(uniq_List)
        returnMatches = uniq_List2.intersection(manuelUniqList2)
        print(' '.join(str(a) for a in returnMatches)) 
                        




        for trakcer_id, bbox in zip(t_ids, bboxes):
            if trakcer_id not in track_id_dict:               
                """
                Eşitlemeye çalıştım ama ID yi görmediği zaman sistemi kapatıyor. 
                Amacım her yeni ID yi detect ettiğinde tracker_ıd ye onu atması
                Bu yüzden aşağıda bir eşitleme kurmaya çalıştım. İki listeyi karşılaştırıp Uniq ID yi alması için.
                """
                new_id = ' '.join(str(a) for a in returnMatches)                     
                track_id_dict[trakcer_id] = new_id
            
          
                
        elapsed_time = time.time() - start_time


        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            score_th,
            t_ids,
            t_bboxes,
            t_scores,
            t_class_ids,
            track_id_dict,
            coco_classes,
        )


        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
            
        cv2.imshow('YOLOX ByteTrack(Multi Class) Sample', debug_image)
        cv2.imshow('AprilTag Detect Demo', debug_image2)

    cap.release()
    cv2.destroyAllWindows()


def get_id_color(index):
    temp_index = abs(int(index)) * 3
    color = ((37 * temp_index) % 255, (17 * temp_index) % 255,
             (29 * temp_index) % 255)
    return color


def draw_debug(
    image,
    elapsed_time,
    score_th,
    trakcer_ids,
    bboxes,
    scores,
    class_ids,
    track_id_dict,
    coco_classes,
):
    debug_image = copy.deepcopy(image)

    for tracker_id, bbox, score, class_id in zip(trakcer_ids, bboxes, scores,
                                                 class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if score_th > score:
            continue

        color = get_id_color(int(track_id_dict[tracker_id]))
        
        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            color,
            thickness=2,
        )

 
        score_txt = str(round(score, 2))
        text = 'Track ID:%s(%s)' % (int(track_id_dict[tracker_id]), score_txt)
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness=2,
        )

        text = 'Class ID:%s(%s)' % (class_id, coco_classes[class_id])
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness=2,
        )


    text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000)
    text = text + 'ms'
    debug_image = cv2.putText(
        debug_image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    return debug_image

def draw_tags(
    image2,
    tags,
    elapsed_time,
):
    for tag in tags:
        tag_family = tag.tag_family
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners

        center = (int(center[0]), int(center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))

        
        cv2.circle(image2, (center[0], center[1]), 5, (0, 0, 255), 2)

       
        cv2.line(image2, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv2.line(image2, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv2.line(image2, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv2.line(image2, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)

        
        cv2.putText(image2, str(tag_id), (center[0] - 10, center[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        

 
    cv2.putText(image2,
               "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv2.LINE_AA)

    return image2
    
     
  
if __name__ == '__main__':
    main()
