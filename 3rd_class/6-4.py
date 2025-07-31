import sensor, image, time, lcd
from maix import KPU
from maix import GPIO
from fpioa_manager import fm
from machine import Timer, PWM
import gc

# ──reset ──────────────────────────────────────────────────────────────
gc.collect()
lcd.init()
sensor.reset()
sensor.set_vflip(1)
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)          # 320×240
sensor.skip_frames(time=1000)
clock = time.clock()

# YOLO anchor
anchor = (0.8125, 0.4556, 1.1328, 1.2667,
          1.8594, 1.4889, 1.4844, 2.2000,
          2.6484, 2.9333)

# KPU & YOLO reset
kpu = KPU()
kpu.load_kmodel("/sd/model/hand_detect.kmodel")
kpu.init_yolo2(anchor,
               anchor_num=5,
               img_w=320, img_h=240,
               net_w=320, net_h=240,    # 256 → 240
               layer_w=10, layer_h=8,
               threshold=0.7, nms_value=0.3,
               classes=1)

# ── motor PWM set ──────────────────────────────────────────────────────
tim0 = Timer(Timer.TIMER0, Timer.CHANNEL0, mode=Timer.MODE_PWM)
tim1 = Timer(Timer.TIMER0, Timer.CHANNEL1, mode=Timer.MODE_PWM)
tim2 = Timer(Timer.TIMER0, Timer.CHANNEL2, mode=Timer.MODE_PWM)
tim3 = Timer(Timer.TIMER0, Timer.CHANNEL3, mode=Timer.MODE_PWM)

left_ia  = PWM(tim0, freq=2000, duty=0, pin=13)
left_ib  = PWM(tim1, freq=2000, duty=0, pin=19)
right_ia = PWM(tim2, freq=2000, duty=0, pin=10)
right_ib = PWM(tim3, freq=2000, duty=0, pin=17)

def car_go(speed):    left_ia.duty(0);      left_ib.duty(speed); right_ia.duty(speed); right_ib.duty(0)
def car_back(speed):  left_ia.duty(speed);  left_ib.duty(0);     right_ia.duty(0);     right_ib.duty(speed)
def car_left(speed):  left_ia.duty(speed);  left_ib.duty(0);     right_ia.duty(speed); right_ib.duty(0)
def car_right(speed): left_ia.duty(0);      left_ib.duty(speed); right_ia.duty(0);     right_ib.duty(speed)

last_time = 0  # ‘stop’ 간격 체크용

# ── main loop ──────────────────────────────────────────────────────────
try:
    while True:
        clock.tick()
        img = sensor.snapshot()
        img.pix_to_ai()                          # od_img 제거, img 직접 사용
        kpu.run_with_output#(img)
        dect = kpu.regionlayer_yolo2()
        fps  = clock.fps()

        if len(dect) > 0:
            for l in dect:
                x = int(l[0] + l[2] / 2)
                y = int(l[1] + l[3] / 2)
                img.draw_cross(x, y, color=(255, 0, 0))
                img.draw_rectangle(l[0], l[1], l[2], l[3], color=(0, 255, 0))
                print("x:{}  y:{}".format(x, y))

                # handling the direction
                if x < 100:
                    print("left")
                    car_left(50)
                elif x > 200:
                    print("right")
                    car_right(50)
                else:
                    print("go")
                    car_go(50)


            del dect

        # FPS display
        img.draw_string(0, 0, "%2.1ffps" % fps, color=(0, 60, 128), scale=2.0)
        lcd.display(img)

        #stop code
        current_time = time.ticks_ms()
        if current_time - last_time >= 1000:
            print("stop")
            car_go(0)
            last_time = current_time

        del img
        if gc.mem_free() < 80 * 1024:
            gc.collect()

except Exception as e:
    print("Error:", e)

finally:
    kpu.deinit()
