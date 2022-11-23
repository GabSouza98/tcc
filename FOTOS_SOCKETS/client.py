import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('192.168.100.11', 12345))


BUFFER_SIZE = 4096

with open('./client_file.png', 'rb') as file:         #ABRE A FOTO DE INTERESSE COMO READ BINARY
    file_data = file.read(BUFFER_SIZE)                #LÊ A FOTO NO TAMANHO DO BUFFER  

    while file_data:                                  #ENVIA PARA O SERVER ATÉ ACABAR OS CHUNKS        
        client.send(file_data) 
        file_data = file.read(BUFFER_SIZE)


client.send(b"%IMAGE_COMPLETED%")                     #ENVIA UM MARCADOR PARA INDICAR FIM DA TRASNFERÊNCIA  

with open('./client_file_edited.png', 'wb') as file:  #RECEBE DO SERVER O STREAM
    recv_data = client.recv(BUFFER_SIZE)              #RECEBE O CHUNK DO SERVER

    while recv_data:                                 #RECEBE ATÉ ACABAR OS CHUNKS        
        file.write(recv_data)        
        recv_data = client.recv(BUFFER_SIZE)

        if recv_data == b"%IMAGE_COMPLETED%":        #VERIFICA SE FOI RECEBIDO O SINAL DE QUE ACABOU A IMAGEM
            recv_data = client.recv(BUFFER_SIZE).decode('utf-8') #PEDE PARA RECEBER MAIS UM PACOTE, E DECODIFICA
            print(recv_data) 
            break
    

        

