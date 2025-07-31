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

od_img = image.Image(size=(320,256))
anchor = (0.8125, 0.4556, 1.1328, 1.2667, 1.8594, 1.4889, 1.4844, 2.2000, 2.6484, 2.9333)

kpu = KPU()
kpu.load_kmodel("/sd/model/hand_detect.kmodel")
kpu.init_yolo2(anchor, anchor_num=5, img_w=320, img_h=240, net_w=320 , net_h=256 ,layer_w=10 ,layer_h=8, threshold=0.7, nms_value=0.3, classes=1)

last_time = 0

while True:
    gc.collect()
    clock.tick()
    img = sensor.snapshot()
    od_img.draw_image(img, 0,0)
    od_img.pix_to_ai()
    kpu.run_with_output(od_img)
    dect = kpu.regionlayer_yolo2()
    fps = clock.fps()

    if len(dect) > 0:
        #print("dect:",dect)
        for l in dect:
            x = int(l[0]+l[2]/2)
            y = int(l[1]+l[3]/2)
            img.draw_cross(x,y, color=(255, 0, 0))
            print("x:{}  y:{}".format(x,y))
            img.draw_rectangle(l[0],l[1],l[2],l[3], color=(0, 255, 0))
            # if x < 100:
            #     print("left")
            # elif x > 200:
            #     print("right")
    # else:
    #           print("go")
    #           car_go(50)

    img.draw_string(0, 0, "%2.1ffps" %(fps), color=(0, 60, 128), scale=2.0)
    lcd.display(img)

    # 현재 시간을 가져옵니다.
    current_time = time.ticks_ms()

    # 일정 시간마다 "stop" 출력
    if current_time - last_time >= 1000:
        print("stop")
        last_time = current_time

kpu.deinit()
