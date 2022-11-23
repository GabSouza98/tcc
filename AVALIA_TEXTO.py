#texto = "HAL : 021221 PET-PCR\nL-P8121610: 002\n"
import re

def verifica_conformidade_fruki(texto):

    texto = texto.split('\n') 

    x = re.findall("[JH¥VvuUW][APR][L]\s?[:]?[']?[·]?[.]?[-]?\s?[0-9]{6}\s?[a-zA-Z]{3}", str(texto[0]).replace(" ",""))  

    if x:
        qualidade_primeira_linha = True
    else:
        qualidade_primeira_linha = False   
   
    #SEGUNDA LINHA                           
    if len(texto)>1:
        y = re.findall("[PF][APR][EB][0-9]{10}", str(texto[1]).replace(" ",""))
        if y:
            qualidade_segunda_linha = True
        else:
            qualidade_segunda_linha = False                          
            
    if(qualidade_primeira_linha and qualidade_segunda_linha):
        #print("Codificação Conforme")
        return "Conforme"
    else:
        #print("Codificação Não Conforme")
        return "Não Conforme"

def verifica_conformidade_pepsi(texto):

    texto = texto.split('\n') 

    x = re.findall("[JH¥VvuUW][APR][L]\s?[:]?[']?[·]?[.]?[-]?\s?[0-9]{6}\s?[PF][EB][TI7][-]?[PF][CO][PRA]", str(texto[0]).replace(" ",""))  

    if x:
        qualidade_primeira_linha = True
    else:
        qualidade_primeira_linha = False   
   
    #SEGUNDA LINHA                           
    if len(texto)>1:
        y = re.findall("[L]\s?[:]?[-]?[']?[·]?[.]?\s?[PF][S839]\s?[0-9]{4}\s?[0-9]{2}\s?[:]?[-]?[']?[·]?[.]?\s?[0-9]{3}", str(texto[1]).replace(" ",""))
        if y:
            qualidade_segunda_linha = True
        else:
            qualidade_segunda_linha = False                          
            
    if(qualidade_primeira_linha and qualidade_segunda_linha):
        #print("Codificação Conforme")
        return "Conforme"
    else:
        #print("Codificação Não Conforme")
        return "Não Conforme"

