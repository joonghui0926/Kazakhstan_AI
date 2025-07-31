import sensor, image, time, lcd
from maix import KPU
import gc
gc.collect()

sensor.reset()
sensor.set_vflip(1)
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QQVGA)
sensor.skip_frames(time = 500)

lcd.init()

#인공지능 모델 불러오기
kpu = KPU()
kpu.load_kmodel("/sd/model/uint8_mnist_cnn_model.kmodel")

while True:
    # remodeling the input data for using the AI model
    gc.collect()
    img = sensor.snapshot() # load the picture
    img_mnist=img.to_grayscale(1) # change to grayscale
    img_mnist=img_mnist.resize(112,112) # resize img 112x112
    img_mnist.invert() # inverting img(most of the picture usally black background with white word)
    img_mnist.strech_char(1) # enlarge the number part for better classfication
    img_mnist.pix_to_ai() # change to model for fitting

    # produce the result
    out = kpu.run_with_output(img_mnist, getlist=True) # process the model by input data
    max_mnist = max(out) # select the maximum logit
    index_mnist = out.index(max_mnist) # choose the finding result
    score = KPU.sigmoid(max_mnist) # show the score
    # select the the result when the score is over 0.99999
    if score >= 0.99999:
        display_str = "number: {} ,score: {}".format(index_mnist,score)
    else:
        display_str = "None"
    print(display_str)
    img.draw_string(4,3,display_str,color=(0,0,0),scale=1)
    lcd.display(img)

# 생성한 KPU 객체 초기화 및 모델 메모리 해제
kpu.deinit()
