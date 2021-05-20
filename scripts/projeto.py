#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function, division
import rospy
import numpy as np
import numpy
import tf
import math
import cv2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from numpy import linalg
from tf import transformations
from tf import TransformerROS
import tf2_ros
from geometry_msgs.msg import Twist, Vector3, Pose, Vector3Stamped

from nav_msgs.msg import Odometry
from std_msgs.msg import Header

"""
print("EXECUTE ANTES da 1.a  vez: ")
print("wget https://github.com/Insper/robot21.1/raw/main/projeto/ros_projeto/scripts/MobileNetSSD_deploy.caffemodel")
print("PARA TER OS PESOS DA REDE NEURAL")
"""

import visao_module


bridge = CvBridge()

cv_image = None
media = []
centro = []
atraso = 1.5E9 # 1 segundo e meio. Em nanossegundos


area = 0.0 # Variavel com a area do maior contorno

# Só usar se os relógios ROS da Raspberry e do Linux desktop estiverem sincronizados. 
# Descarta imagens que chegam atrasadas demais
check_delay = False 

resultados = [] # Criacao de uma variavel global para guardar os resultados vistos

x = 0
y = 0
z = 0 
id = 0

frame = "camera_link"
# frame = "head_camera"  # DESCOMENTE para usar com webcam USB via roslaunch tag_tracking usbcam

tfl = 0

tf_buffer = tf2_ros.Buffer()

distancia = 0

zero = Twist(Vector3(0,0,0), Vector3(0,0,0))         

def scaneou(dado):

    global distancia
    
    ranges = np.array(dado.ranges).round(decimals=2)
    distancia = ranges[0]
    lateral_direita = ranges[1]

# A função a seguir é chamada sempre que chega um novo frame
def roda_todo_frame(imagem):
    print("frame")
    global cv_image
    global media
    global centro
    global resultados

    now = rospy.get_rostime()
    imgtime = imagem.header.stamp
    lag = now-imgtime # calcula o lag
    delay = lag.nsecs
    # print("delay ", "{:.3f}".format(delay/1.0E9))
    if delay > atraso and check_delay==True:
        # Esta logica do delay so' precisa ser usada com robo real e rede wifi 
        # serve para descartar imagens antigas
        print("Descartando por causa do delay do frame:", delay)
        return 
    try:
        temp_image = bridge.compressed_imgmsg_to_cv2(imagem, "bgr8")
        # Note que os resultados já são guardados automaticamente na variável
        # chamada resultados
        centro, saida_net, resultados =  visao_module.processa(temp_image)        
        for r in resultados:
            # print(r) - print feito para documentar e entender
            # o resultado            
            pass

        # Desnecessário - Hough e MobileNet já abrem janelas
        cv_image = saida_net.copy()
        cv2.imshow("cv_image", cv_image)
        cv2.waitKey(1)
    except CvBridgeError as e:
        print('ex', e)

def recebe_odometria(data):
    x = data.pose.pose.orientation.x
    y = data.pose.pose.orientation.y
    print("Valores da odometria:", x,y)
    
def segmenta_linha_amarela(bgr):    
    """Não mude ou renomeie esta função
        deve receber uma imagem bgr e retornar os segmentos amarelos do centro da pista em branco.
        Utiliza a função cv2.morphologyEx() para limpar ruidos na imagem
    """
    img_hsv = cv2.cvtColor(bgr.copy(), cv2.COLOR_BGR2HSV)

    hsv1 = np.array([20, 100, 150])
    hsv2 = np.array([35, 255, 255])

    mask = cv2.inRange(img_hsv, hsv1, hsv2)

    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5, 5)))

    return mask

def encontrar_contornos(mask):
    """Não mude ou renomeie esta função
        deve receber uma imagem preta e branca os contornos encontrados
    """
    
    
    contornos, arvore = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contornos
    

    

def crosshair(img, point, size, color):
    """ Desenha um crosshair centrado no point.
        point deve ser uma tupla (x,y)
        color é uma tupla R,G,B uint8
    """
    x,y = point
    cv2.line(img,(x - size,y),(x + size,y),color,2)
    cv2.line(img,(x,y - size),(x, y + size),color,2)

def encontrar_centro_dos_contornos(img, contornos):
    """Não mude ou renomeie esta função
        deve receber um contorno e retornar, respectivamente, a imagem com uma cruz no centro de cada segmento e o centro dele. formato: img, x, y
    """
    img_contornos = img.copy()
    X = []
    Y = []

    for contorno in contornos:
        M = cv2.moments(contorno)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            point = (int(cX), int(cY))
            crosshair(img_contornos, point, 15, (0, 0, 255))
            X.append(int(cX))
            Y.append(int(cY))
        except:
            pass

    return img_contornos, X, Y

def desenhar_linha_entre_pontos(img, X, Y, color):
    """Não mude ou renomeie esta função
        deve receber uma lista de coordenadas XY, e retornar uma imagem com uma linha entre os centros EM SEQUENCIA do mais proximo.
    """
    linha = img.copy()
    for i in range(len(X)-1):
        cv2.line(linha,(X[i],Y[i]),(X[i+1],Y[i+1]),(255, 0, 0), 2)

    return linha

def regressao_por_centro(img, x,y):
    """Não mude ou renomeie esta função
        deve receber uma lista de coordenadas XY, e estimar a melhor reta, utilizando o metodo preferir, que passa pelos centros. Retorne a imagem com a reta e os parametros da reta

        Dica: cv2.line(img,ponto1,ponto2,color,2) desenha uma linha que passe entre os pontos, mesmo que ponto1 e ponto2 não pertençam a imagem.
    """
    
    value =[]
    j = 0
    while j < len(x):
        a = ((x[j]*x[j]) + (y[j]*y[j]))**(1/2)
        value.append(a)
        j+=1
    # First quartile (Q1) 
    Q1 = np.percentile(value, 25, interpolation = 'midpoint') 
    
    # Third quartile (Q3) 
    Q3 = np.percentile(value, 75, interpolation = 'midpoint') 
    
    # Interquaritle range (IQR) 
    iqr = Q3 - Q1
    filterx = []
    filtery = []
    i=0
    while i < len(value)-1:
        if Q1-(iqr*1.5) <= value[i]:
            if Q3+(iqr*1.5) >= value[i]:
                filterx.append(x[i])
                filtery.append(y[i])
                i += 1
        else: 
            i+=1
        
    regressao = img.copy()
    slope, intercept, r, p, std_err = stats.linregress(filterx, filtery)

    ponto1 = (img.shape[1], int(slope * img.shape[1] + intercept))
    ponto2 = (0, int(slope * 0 + intercept))

    cv2.line(regressao, ponto1, ponto2, (0, 255, 0), 10)

    return regressao, [slope, intercept]

def calcular_angulo_com_vertical(img, lm):
    """Não mude ou renomeie esta função
        deve receber uma lista de coordenadas XY, e estimar a melhor reta, utilizando o metodo preferir, que passa pelos centros. Retorne a imagem com a reta.

        Dica: cv2.line(img,ponto1,ponto2,color,2) desenha uma linha que passe entre os pontos, mesmo que ponto1 e ponto2 não pertençam a imagem.
    """
    m = lm[0]

    alpha = math.atan(m)
    beta = abs(math.degrees(alpha))
    angulo = 90 - beta

    return angulo

if __name__=="__main__":
    rospy.init_node("cor")

    topico_imagem = "/camera/image/compressed"

    recebedor = rospy.Subscriber(topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size = 2**24)
    recebe_odom = rospy.Subscriber("/odom", Odometry , recebe_odometria)


    print("Usando ", topico_imagem)

    velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

    tfl = tf2_ros.TransformListener(tf_buffer) #conversao do sistema de coordenadas 
    tolerancia = 25

    try:
        # Inicializando - por default gira no sentido anti-horário
        vel = Twist(Vector3(0,0,0), Vector3(0,0,math.pi/10.0))
        
        while not rospy.is_shutdown():
            for r in resultados:
                print(r)
            
            velocidade_saida.publish(vel)
            rospy.sleep(0.1)

    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")


