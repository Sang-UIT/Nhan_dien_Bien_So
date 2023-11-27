from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
import torch
import function.utils_rotate as utils_rotate
import function.helper as helper
# load model
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

class Detect(VideoTransformerBase):
    def __init__(self):
        self.i = 0
    def transform(self,frame):
        img = frame.to_ndarray(format="bgr24")
        plates = yolo_LP_detect(img, size=640)
        list_plates = plates.pandas().xyxy[0].values.tolist()
        list_read_plates = set()
        for plate in list_plates:
            flag = 0
            x = int(plate[0]) # xmin
            y = int(plate[1]) # ymin
            w = int(plate[2] - plate[0]) # xmax - xmin
            h = int(plate[3] - plate[1]) # ymax - ymin  
            crop_img = img[y:y+h, x:x+w]
            cv2.rectangle(img, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
            cv2.imwrite("crop.jpg", crop_img)
            rc_image = cv2.imread("crop.jpg")
            lp = ""
            for cc in range(0,2):
                for ct in range(0,2):
                    lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(rc_image, cc, ct))
                    if lp != "unknown":
                        list_read_plates.add(lp)
                        cv2.putText(img, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        flag = 1
                    break
            if flag == 1:
                break
        return img