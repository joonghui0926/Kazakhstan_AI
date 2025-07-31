import sensor, image, time, lcd
from maix import KPU
from maix import GPIO
from fpioa_manager import fm
from board import board_info
from machine import Timer, PWM
import gc

# ── initialize ──────────────────────────────────────────────────────────────
gc.collect()
lcd.init()
sensor.reset()
sensor.set_vflip(1)
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)          # 320×240 lower the resolution
sensor.skip_frames(time=1000)
clock = time.clock()

# VOC-20 class & anker
obj_name = ("aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
            "chair","cow","diningtable","dog","horse","motorbike","person",
            "pottedplant","sheep","sofa","train","tvmonitor")
anchor = (1.3221,1.73145, 3.19275,4.00944, 5.05587,8.09892,
          9.47112,4.84053, 11.2364,10.0071)

# KPU & YOLO initialize
kpu = KPU()
kpu.load_kmodel("/sd/model/voc20_detect.kmodel")
kpu.init_yolo2(anchor,
               anchor_num=5,
               img_w=320, img_h=240,
               net_w=320, net_h=240,      # 256 → 240
               layer_w=10, layer_h=8,
               threshold=0.5, nms_value=0.2,
               classes=20)

# ── Set moter PWM ──────────────────────────────────────────────────────
tim0 = Timer(Timer.TIMER0, Timer.CHANNEL0, mode=Timer.MODE_PWM)
tim1 = Timer(Timer.TIMER0, Timer.CHANNEL1, mode=Timer.MODE_PWM)
tim2 = Timer(Timer.TIMER0, Timer.CHANNEL2, mode=Timer.MODE_PWM)
tim3 = Timer(Timer.TIMER0, Timer.CHANNEL3, mode=Timer.MODE_PWM)

left_ia  = PWM(tim0, freq=2000, duty=0, pin=13)
left_ib  = PWM(tim1, freq=2000, duty=0, pin=19)
right_ia = PWM(tim2, freq=2000, duty=0, pin=10)
right_ib = PWM(tim3, freq=2000, duty=0, pin=17)

def car_go(s):    left_ia.duty(0);    left_ib.duty(s); right_ia.duty(s); right_ib.duty(0)
def car_back(s):  left_ia.duty(s);    left_ib.duty(0); right_ia.duty(0); right_ib.duty(s)
def car_left(s):  left_ia.duty(s);    left_ib.duty(0); right_ia.duty(s); right_ib.duty(0)
def car_right(s): left_ia.duty(0);    left_ib.duty(s); right_ia.duty(0); right_ib.duty(s)

def car_stop(t):
    car_go(0)
    print("멈춤")

# timer will stop in every 0.5sec
tim = Timer(Timer.TIMER1, Timer.CHANNEL0, mode=Timer.MODE_PERIODIC,
            period=500, callback=car_stop)

# ── main ──────────────────────────────────────────────────────────
try:
    while True:
        clock.tick()
        img = sensor.snapshot()          # QVGA(320×240)
        img.pix_to_ai()                  # r8g8b8 format
        kpu.run_with_output(img)
        dect = kpu.regionlayer_yolo2()
        fps  = clock.fps()

        if len(dect):
            for l in dect:
                cls = obj_name[l[4]]
                if cls == "cat":         # track cat
                    x = int(l[0] + l[2]/2)
                    y = int(l[1] + l[3]/2)
                    img.draw_rectangle(l[0], l[1], l[2], l[3], color=(0,255,0))
                    img.draw_string(l[0], l[1], cls, color=(0,255,0), scale=1.5)
                    img.draw_cross(x, y)

                    if x < 100:
                        print("left", x, y)
                        car_left(50)
                    elif x > 200:
                        print("right", x, y)
                        car_right(50)
                    else:
                        print("straight", x, y)
                        car_go(50)
        # FPS·LCD
        img.draw_string(0, 0, "%2.1ffps" % fps, color=(0,60,128), scale=1.0)
        lcd.display(img)

        # ── managing memory ────────────────────────────────────────────────
        del dect
        del img
        if gc.mem_free() < 80 * 1024:    # free < 80 KB -> GC
            gc.collect()

except Exception as e:
    print("Error:", e)

finally:
    kpu.deinit()
