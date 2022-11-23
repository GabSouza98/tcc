import io
import socket
from time import sleep
from PIL import Image, ImageFilter


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

PORT = 12345
IP = '192.168.100.11'
server.bind((IP, PORT))   #especifica que vai rodar no localhost
server.listen(1) # servidor escutando
print("LISTENING")

BUFFER_SIZE = 4096

while True:
    client_socket, addr = server.accept()  #aceita clientes

    print("CONEXAO ESTABELECIDA")
    
    with open('./client_file.png', 'rb') as file:         #ABRE A FOTO DE INTERESSE COMO READ BINARY
        file_data = file.read(BUFFER_SIZE)                #LÊ A FOTO NO TAMANHO DO BUFFER  

        while file_data:                                  #ENVIA PARA O SERVER ATÉ ACABAR OS CHUNKS        
            client_socket.send(file_data)             
            file_data = file.read(BUFFER_SIZE)

        print("fechando arquivo")

    print("arquivo fechado") 
    sleep(2)
    client_socket.send(b"%IMAGE_COMPLETED%")                     #ENVIA UM MARCADOR PARA INDICAR FIM DA TRASNFERÊNCIA  
    
    
    print("AGUARDANDO RESULTADO DO CLIENTE")

    client_socket.send(b"%IMAGE_COMPLETED%") 
    
    recv_data = client_socket.recv(BUFFER_SIZE).decode('utf-8')
    print(recv_data)             
    
