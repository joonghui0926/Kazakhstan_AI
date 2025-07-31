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
    img = sensor.snapshot()             # bring the picture 촬영하여 이미지 가져오기
    od_img.draw_image(img, 0,0)     # draw the img at (0,0) 이미지를 od_img 이미지의 (0,0) 위치에 그립니다.
    od_img.pix_to_ai()                  # change the image to fit the model rgb565 이미지를 AI 연산에 필요한 r8g8b8 형식으로 변환
    kpu.run_with_output(od_img)         # caluate the input image 입력 이미지에 KPU 연산 수행
    dect = kpu.regionlayer_yolo2()      # process it after YOLO2 YOLO2 후 처리
    fps = clock.fps()                   # bring FPS FPS 가져오기

    # draw boundary box and present class of object  박스 그리기 및 객체 클래스 표시
    if len(dect) > 0:
   
        for l in dect :
            #if obj_name[l[4]] == "cat":  # dectect the only for "cat"
                #img.draw_rectangle(l[0],l[1],l[2],l[3], color=(0, 255, 0)) # draw the boundary box
                #img.draw_string(l[0],l[1], obj_name[l[4]], color=(0, 255, 0), scale=1.5)
                #x = int(l[0]+l[2]/2)
                #y = int(l[1]+l[3]/2)
                #img.draw_cross(x,y) #find the center of box
                #print("x:{}  y:{}".format(x,y))
    img.draw_string(0, 0, "%2.1ffps" %(fps), color=(0, 60, 128), scale=1.0)
    lcd.display(img)
    gc.collect()

# KPU 객체 초기화 및 모델 메모리 해제
kpu.deinit()

# 찾은 고양이의 중심좌표 구하기
