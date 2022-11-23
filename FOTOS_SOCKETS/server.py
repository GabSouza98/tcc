import io
import socket
from PIL import Image, ImageFilter

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

PORT = 12346
IP = '192.168.100.11'
server.bind((IP, PORT))   #especifica que vai rodar no localhost
server.listen() # servidor escutando
print("LISTENING")

BUFFER_SIZE = 4096

while True:
    client_socket, _ = server.accept()  #aceita clientes

    print("CONEXAO ESTABELECIDA")
    file_stream = io.BytesIO()                                         # CRIA UM BYTESREAM
    recv_data = client_socket.recv(BUFFER_SIZE)                        # RECEBE O PRIMEIRO CHUNK DO CLIENT
  
    while recv_data:                                                   # ENQUANTO NÃO RECEBER VAZIO
        print("RECEBENDO DADOS")
        file_stream.write(recv_data)                                   # ESCREVE O CHUNK RECEBIDO NO FILE_STREAM
        recv_data = client_socket.recv(BUFFER_SIZE)                    # PEDE PARA RECEBER O PRÓXIMO CHUNK

        if recv_data == b"%IMAGE_COMPLETED%":                           # VERIFICAR SE CHEGOU O MARCADOR DE FIM  
            print("IMAGEM RECEBIDA")          
            break

    #################################################################### TRATAMENTOS
    image = Image.open(file_stream)                                    
    image = image.filter(ImageFilter.GaussianBlur(radius=10))
    image.save('./server_file.png')    
    #################################################################### FIM TRATAMENTOS

    with open('./server_file.png', 'rb') as file:                      # ABRE O ARQUIVO TRATADO EM BINARIO
        file_data = file.read(BUFFER_SIZE)                             # COMEÇA A PASSAR O STREAM
        while file_data:                                               # ENQUANTO NÃO TERMINAR O ARQUIVO     
            print("ENVIANDO DADOS")       
            client_socket.send(file_data)
            file_data = file.read(BUFFER_SIZE)

    client_socket.send(b"%IMAGE_COMPLETED%")                           # ENVIA O MARCADOR DE TÉRMINO DE TRANSF
    print("IMAGEM ENVIADA")
    msg = "DEU TUDO CERTO".encode('utf-8')
    client_socket.send(msg)
