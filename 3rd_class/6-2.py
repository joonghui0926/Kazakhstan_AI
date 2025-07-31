import sensor, image, time, lcd
from maix import KPU
import gc
gc.collect()  # 메모리 해제

lcd.init()
sensor.reset()
sensor.set_vflip(1)
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time = 1000)
clock = time.clock()


od_img = image.Image(size=(320,256))  # 320x256 크기의 이미지 객체 초기화

# define the object name and anchor
obj_name = ("aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog","horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep","sofa",
            "train",
            "tvmonitor")
anchor = (1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071)

# load model
kpu = KPU()
kpu.load_kmodel("/sd/model/voc20_detect.kmodel")

# reset YOLO2
kpu.init_yolo2(anchor, anchor_num=5, img_w=320, img_h=240, net_w=320 , net_h=256 ,layer_w=10 ,layer_h=8, threshold=0.5, nms_value=0.2, classes=20)


while True:
    clock.tick()                        # update the frame speed 프레임 속도 계산 업데이트
    img = #sensor.snapshot()             # bring the picture 촬영하여 이미지 가져오기
    od_img.draw_image(img, 0,0)     # draw the image at (0,0) 이미지를 od_img 이미지의 (0,0) 위치에 그립니다.
    od_img.pix_to_ai()                  # change the image to fit the model rgb565 이미지를 AI 연산에 필요한 r8g8b8 형식으로 변환
    kpu.run_with_output(od_img)         # caluate the input image 입력 이미지에 KPU 연산 수행
    dect = #kpu.regionlayer_yolo2()      # process it after YOLO2 YOLO2 후 처리
    fps = #clock.fps()                   # bring FPS  FPS 가져오기
    print(dect)

    lcd.display(img)

    gc.collect()

# KPU 객체 초기화 및 모델 메모리 해제
kpu.deinit()

# 검출한 객체 출력하기
