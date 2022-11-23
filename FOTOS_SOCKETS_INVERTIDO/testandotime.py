import time

start_time = time.time()

time.sleep(4)

i = 0
soma=0
while (i<20):
    soma=soma+i
    i=i+1
    print(soma)


tempo_total = "--- %s seconds ---" % round(time.time() - start_time, 3)
print(tempo_total)